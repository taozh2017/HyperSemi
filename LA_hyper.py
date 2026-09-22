import os
import sys
from xml.etree.ElementInclude import default_loader
from tqdm import tqdm
import shutil
import argparse
import logging
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
import hypersemi.utils.losses as losses

from yaml import parse
from skimage.measure import label
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils import ramps,test_3d_patch
from dataloaders.LADataset import LAHeart
from utils.LA_utils import to_cuda
from utils.BCP_utils import *
from networks.Vnet import VNet
from networks.ResVNet import ResVNet
from dataloaders.dataloader import get_network
import time

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./Datasets/la/data', help='Name of Dataset')
parser.add_argument('--exp', type=str, default='hyperSemi', help='exp_name')
parser.add_argument('--model', type=str, default='VNet', help='model_name')
parser.add_argument('--pre_max_iteration', type=int, default=2000, help='maximum pre-train iteration to train')
parser.add_argument('--self_max_iteration', type=int, default=15000, help='maximum self-train iteration to train')
parser.add_argument('--max_samples', type=int, default=80, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=4, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=1e-3, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=8, help='trained samples')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--seed', type=int, default=1345, help='random seed')
parser.add_argument('--consistency', type=float, default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float, default='10.0', help='magnitude')
# -- setting of BCP
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
parser.add_argument('--mask_ratio', type=float, default=2 / 3, help='ratio of mask/image')
# -- setting of mixup
parser.add_argument('--u_alpha', type=float, default=2.0, help='unlabeled image ratio of mixuped image')
parser.add_argument('--loss_weight', type=float, default=0.5, help='loss weight of unimage term')

# -- setting of CoOP
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--n_ctx', type=int, default=4, help='prompt max len')
parser.add_argument('--dataset', type=str, default='LA', help='Name of Experiment')
args = parser.parse_args()


def create_Vnet(ema=False):
    net = VNet(n_channels=1, n_classes=2, normalization='instancenorm', has_dropout=True)
    net = nn.DataParallel(net)
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def create_ResVnet(ema=False):
    net = ResVNet(n_channels=1, n_classes=2, normalization='instancenorm', has_dropout=True)
    net = nn.DataParallel(net)
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def get_cut_mask(out, thres=0.5, nms=0):
    probs = F.softmax(out, 1)
    masks = (probs >= thres).type(torch.int64)
    masks = masks[:, 1, :, :].contiguous()
    if nms == 1:
        masks = LargestCC_pancreas(masks)
    return masks


def get_cut_mask_two(out1, out2, thres=0.5, nms=0):
    probs1 = F.softmax(out1, 1)
    probs2 = F.softmax(out2, 1)
    probs = (probs1 + probs2) / 2

    masks = (probs >= thres).type(torch.int64)
    masks = masks[:, 1, :, :].contiguous()
    if nms == 1:
        masks = LargestCC_pancreas(masks)
    return masks


def LargestCC_pancreas(segmentation):
    N = segmentation.shape[0]
    batch_list = []
    for n in range(N):
        n_prob = segmentation[n].detach().cpu().numpy()
        labels = label(n_prob)
        if labels.max() != 0:
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        else:
            largestCC = n_prob
        batch_list.append(largestCC)

    return torch.Tensor(batch_list).cuda()


def save_net_opt(net, optimizer, path):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
    }
    torch.save(state, str(path))


def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])


def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def get_uncertainty_mask(pred1, pred2, patch_size):
    """
    patch_size: (px, py, pz)
    """
    B, C, H, W, D = pred1.shape
    
    patch_size = tuple(int(p * 2 / 3) for p in patch_size)
    px, py, pz = patch_size

    p1 = torch.softmax(pred1, dim=1)
    p2 = torch.softmax(pred2, dim=1)

    uncertainty_map = torch.mean((p1 - p2) ** 2, dim=1, keepdim=True)


    smooth_map = F.avg_pool3d(
        uncertainty_map,
        kernel_size=(px, py, pz),
        stride=1
    )

    mask = torch.zeros((B, H, W, D), dtype=torch.float32, device=pred1.device)

    for b in range(B):
        sample_map = smooth_map[b, 0]  # (H', W', D')

        flat_idx = torch.argmax(sample_map)
        x, y, z = torch.unravel_index(flat_idx, sample_map.shape)

        x, y, z = x.item(), y.item(), z.item()

        x_end = min(x + px, H)
        y_end = min(y + py, W)
        z_end = min(z + pz, D)

        mask[b, x:x_end, y:y_end, z:z_end] = 1.0

    return mask

def poincare_distance(x, y, c=1.0):
    """
        d(x, y) = (1/sqrt(c)) * arccosh( 1 + delta )
    """
    sqrt_c = c ** 0.5
    
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    
    dist_sq_euclidean = torch.sum((x - y) ** 2, dim=-1, keepdim=True)
    denom_x = 1 - c * x2
    denom_y = 1 - c * y2
    
    denom_x = torch.clamp(denom_x, min=1e-15)
    denom_y = torch.clamp(denom_y, min=1e-15)
    
    alpha = 1 + 2 * c * dist_sq_euclidean / (denom_x * denom_y)
    alpha = torch.clamp(alpha, min=1.0 + 1e-7)
    
    dist = (1.0 / sqrt_c) * torch.acosh(alpha)
    
    return dist

def calc_align_loss(v_hyp, t_hyp, mixed_label, curvature=0.1):

    dist_matrix = poincare_distance(v_hyp.unsqueeze(1), t_hyp.unsqueeze(0), c=curvature).squeeze(-1)
    has_foreground = (mixed_label == 1).sum(dim=(1, 2, 3)) > 0
    target_class = has_foreground.long().to(dist_matrix.device)
    loss = losses.CE(-dist_matrix, target_class).mean()
    return loss

train_data_path = args.root_path

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
pre_max_iterations = args.pre_max_iteration
self_max_iterations = args.self_max_iteration
base_lr = args.base_lr

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

patch_size = (112, 112, 80)
num_classes = 2


def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])


def save_net_opt(net, optimizer, path, epoch):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, str(path))


def get_XOR_region(mixout1, mixout2):
    s1 = torch.softmax(mixout1, dim=1)
    l1 = torch.argmax(s1, dim=1)

    s2 = torch.softmax(mixout2, dim=1)
    l2 = torch.argmax(s2, dim=1)

    diff_mask = (l1 != l2)
    return diff_mask


def cmp_dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def pre_train(args, snapshot_path):

    model, model2, optimizer, optimizer2 = get_network(args)

    c_batch_size = 2
    trainset_lab_a = LAHeart(train_data_path, "./Datasets/la/data_split", split='train_lab', logging=logging)
    lab_loader_a = DataLoader(trainset_lab_a, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)

    trainset_lab_b = LAHeart(train_data_path, "./Datasets/la/data_split", split='train_lab', reverse=True, logging=logging)
    lab_loader_b = DataLoader(trainset_lab_b, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)

    model.train()
    model2.train()
    logging.info("{} iterations per epoch".format(len(lab_loader_a)))
    iter_num = 0
    best_dice = 0
    best_dice2 = 0
    max_epoch = 81
    iterator = tqdm(range(1, max_epoch), ncols=70)
    for epoch_num in iterator:
        logging.info("\n")
        for step, ((img_a, lab_a), (img_b, lab_b)) in enumerate(zip(lab_loader_a, lab_loader_b)):
            img_a, img_b, lab_a, lab_b = img_a.cuda(), img_b.cuda(), lab_a.cuda(), lab_b.cuda()
            with torch.no_grad():
                out_a_1 = model(img_a)[0]
                out_a_2 = model2(img_a)[0]
                
                img_mask = get_uncertainty_mask(out_a_1, out_a_2, patch_size)
                img_mask = img_mask.unsqueeze(1).float()

            curv_1 = model.module.vision_hyp_proj.c if hasattr(model, 'module') else model.vision_hyp_proj.c
            curv_2 = model2.module.vision_hyp_proj.c if hasattr(model2, 'module') else model2.vision_hyp_proj.c
            volume_batch = img_a * img_mask + img_b * (1 - img_mask)
            lab_mask = img_mask.squeeze(1).long() 
            label_batch = lab_a * lab_mask + lab_b * (1 - lab_mask)

            outputs, v_hyp_l_1, t_hyp_1 = model(volume_batch)
            loss = losses.seg_loss(outputs, label_batch)
            loss_1_align = calc_align_loss(v_hyp_l_1, t_hyp_1, label_batch, curvature=curv_1)
            loss = loss + 0.1 * loss_1_align

            outputs2, v_hyp_l_2, t_hyp_2 = model2(volume_batch)
            loss2 = losses.seg_loss(outputs2, label_batch)
            loss_2_align = calc_align_loss(v_hyp_l_2, t_hyp_2, label_batch, curvature=curv_2)
            loss2 = loss2 + 0.1 * loss_2_align

            iter_num += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            logging.info(
                'iteration %d : loss: %03f, loss_dice: %03f, loss_ce: %03f' % (iter_num, loss, loss_dice, loss_ce))

        if epoch_num % 5 == 0:
            model.eval()
            dice_sample = test_3d_patch.var_all_case_LA(model, num_classes=num_classes, patch_size=patch_size,
                                                        stride_xy=18, stride_z=4)
            if dice_sample > best_dice:
                best_dice = round(dice_sample, 4)
                save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                save_best_path = os.path.join(snapshot_path, 'best_model.pth'.format(args.model))
                save_net_opt(model, optimizer, save_mode_path, epoch_num)
                save_net_opt(model, optimizer, save_best_path, epoch_num)
                logging.info("save best model to {}".format(save_mode_path))

            model.train()

            model2.eval()
            dice_sample2 = test_3d_patch.var_all_case_LA(model2, num_classes=num_classes, patch_size=patch_size,
                                                         stride_xy=18, stride_z=4)
            if dice_sample2 > best_dice2:
                best_dice2 = round(dice_sample2, 4)
                save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}_resnet.pth'.format(iter_num, best_dice2))
                save_best_path = os.path.join(snapshot_path, 'best_model_resnet.pth'.format(args.model))
                save_net_opt(model2, optimizer2, save_mode_path, epoch_num)
                save_net_opt(model2, optimizer2, save_best_path, epoch_num)
                logging.info("save best resnet model to {}".format(save_mode_path))
            model2.train()



def self_train(args, pre_snapshot_path, self_snapshot_path):

    model1, model2, optimizer, optimizer2 = get_network(args)

    c_batch_size = 2
    trainset_lab_a = LAHeart(train_data_path, "./Datasets/la/data_split", split='train_lab', logging=logging)
    lab_loader_a = DataLoader(trainset_lab_a, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)

    trainset_lab_b = LAHeart(train_data_path, "./Datasets/la/data_split", split='train_lab', reverse=True, logging=logging)
    lab_loader_b = DataLoader(trainset_lab_b, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)

    trainset_unlab_a = LAHeart(train_data_path, "./Datasets/la/data_split", split='train_unlab', logging=logging)
    unlab_loader_a = DataLoader(trainset_unlab_a, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)

    trainset_unlab_b = LAHeart(train_data_path, "./Datasets/la/data_split", split='train_unlab', reverse=True, logging=logging)
    unlab_loader_b = DataLoader(trainset_unlab_b, batch_size=c_batch_size, shuffle=False, num_workers=0, drop_last=True)

    pretrained_model = os.path.join(pre_snapshot_path, 'best_model.pth')
    pretrained_model2 = os.path.join(pre_snapshot_path, 'best_model_resnet.pth')

    load_net_opt(model1, optimizer, pretrained_model)
    load_net_opt(model2, optimizer2, pretrained_model2)



    model1.train()
    model2.train()

    logging.info("{} iterations per epoch".format(len(lab_loader_a)))
    iter_num = 0
    best_dice = 0
    best_dice2 = 0
    mean_best_dice = 0
    max_epoch = 276
    iterator = tqdm(range(1, max_epoch), ncols=70)
    for epoch in iterator:
        logging.info("\n")

        epoch_start_time = time.time()
        torch.cuda.reset_peak_memory_stats()
        for step, ((img_a, lab_a), (img_b, lab_b), (unimg_a, unlab_a), (unimg_b, unlab_b)) in enumerate(
                zip(lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b)):
            img_a, lab_a, img_b, lab_b, unimg_a, unlab_a, unimg_b, unlab_b = to_cuda(
                [img_a, lab_a, img_b, lab_b, unimg_a, unlab_a, unimg_b, unlab_b])
            with torch.no_grad():

                unoutput_a_1 = model1(unimg_a)[0]
                unoutput_b_1 = model1(unimg_b)[0]
                unoutput_a_2 = model2(unimg_a)[0]
                unoutput_b_2 = model2(unimg_b)[0]


                prob_a_1 = torch.softmax(unoutput_a_1, dim=1)
                prob_a_2 = torch.softmax(unoutput_a_2, dim=1)
                mean_conf_a_1 = prob_a_1.max(dim=1)[0].mean()
                mean_conf_a_2 = prob_a_2.max(dim=1)[0].mean()
                if mean_conf_a_1 > mean_conf_a_2:
                    uimg_a_plab = get_cut_mask(unoutput_a_1, nms=True)
                else:
                    uimg_a_plab = get_cut_mask(unoutput_a_2, nms=True)

                prob_b_1 = torch.softmax(unoutput_b_1, dim=1)
                prob_b_2 = torch.softmax(unoutput_b_2, dim=1)
                mean_conf_b_1 = prob_b_1.max(dim=1)[0].mean()
                mean_conf_b_2 = prob_b_2.max(dim=1)[0].mean()
                if mean_conf_b_1 > mean_conf_b_2:
                    uimg_b_plab = get_cut_mask(unoutput_b_1, nms=True)
                else:
                    uimg_b_plab = get_cut_mask(unoutput_b_2, nms=True)

                
                img_mask_a = get_uncertainty_mask(unoutput_a_1, unoutput_a_2, patch_size)
                img_mask_a = img_mask_a.unsqueeze(1).float()

                img_mask_b = get_uncertainty_mask(unoutput_b_1, unoutput_b_2, patch_size)
                img_mask_b = img_mask_b.unsqueeze(1).float()


            loss_mask_a = img_mask_a.squeeze(1)
            lab_mask_a = loss_mask_a.long()
            loss_mask_b = img_mask_b.squeeze(1)
            lab_mask_b = loss_mask_b.long()
            mixl_img = unimg_a * img_mask_a + img_b * (1 - img_mask_a)
            mixu_img = unimg_b * img_mask_b + img_a * (1 - img_mask_b)
            mixl_label = uimg_a_plab.long() * lab_mask_a + lab_b.long() * (1 - lab_mask_a)
            mxinu_label = uimg_b_plab.long() * lab_mask_b + lab_a.long() * (1 - lab_mask_b)
            curv_1 = model1.module.vision_hyp_proj.c if hasattr(model1, 'module') else model1.vision_hyp_proj.c
            curv_2 = model2.module.vision_hyp_proj.c if hasattr(model2, 'module') else model2.vision_hyp_proj.c


            outputs_l, v_hyp_l_1, t_hyp_l = model1(mixl_img)
            outputs_u, v_hyp_u_1, t_hyp_u = model1(mixu_img)
            loss_l = losses.mix_loss(outputs_l, uimg_a_plab.long(), lab_b, img_mask_a, unlab=True)
            loss_u = losses.mix_loss(outputs_u, uimg_b_plab.long(), lab_a, img_mask_b, unlab=True)
            loss_l_1_align = calc_align_loss(v_hyp_l_1, t_hyp_l, mixl_label, curvature=curv_1)
            loss_u_1_align = calc_align_loss(v_hyp_u_1, t_hyp_u, mxinu_label, curvature=curv_1)

            outputs_l_2, v_hyp_l_2, t_hyp_l_2 = model2(mixl_img)
            outputs_u_2, v_hyp_u_2, t_hyp_u_2 = model2(mixu_img)
            loss_l_2 = losses.mix_loss(outputs_l_2, uimg_a_plab.long(), lab_b, img_mask_a, unlab=True)
            loss_u_2 = losses.mix_loss(outputs_u_2, uimg_b_plab.long(), lab_a, img_mask_b, unlab=True)
            loss_l_2_align = calc_align_loss(v_hyp_l_2, t_hyp_l_2, mixl_label, curvature=curv_2)
            loss_u_2_align = calc_align_loss(v_hyp_u_2, t_hyp_u_2, mxinu_label, curvature=curv_2)

            target_l_1 = outputs_l.detach().argmax(dim=1)
            target_l_2 = outputs_l_2.detach().argmax(dim=1)
            target_u_1 = outputs_u.detach().argmax(dim=1)
            target_u_2 = outputs_u_2.detach().argmax(dim=1)
            consistency_loss = (
                losses.seg_loss(outputs_l, target_l_2, loss_mask_a)
                + losses.seg_loss(outputs_l_2, target_l_1, loss_mask_a)
                + losses.seg_loss(outputs_u, target_u_2, loss_mask_b)
                + losses.seg_loss(outputs_u_2, target_u_1, loss_mask_b)
            )
            consistency_weight = 0.1 * np.exp(-5 * (1 - iter_num / self_max_iterations) ** 2)

            loss = (loss_l + loss_u) + 0.1 * (loss_l_1_align + loss_u_1_align)

            loss_2 = (loss_l_2 + loss_u_2) + 0.1 * (loss_l_2_align + loss_u_2_align)

            total_loss = loss + loss_2 + consistency_weight * consistency_loss

            iter_num += 1

            optimizer.zero_grad()
            optimizer2.zero_grad()
            total_loss.backward()
            optimizer.step()
            optimizer2.step()

            logging.info('epoch %d iteration %d : loss: %03f, loss_l: %03f, loss_u: %03f, loss_l_1_align: %f, loss_u_1_align: %f, loss_l_2_align: %f, loss_u_2_align: %f, consistency_loss: %f \
               ' % (epoch, iter_num, loss, loss_l, loss_u, loss_l_1_align, loss_u_1_align, loss_l_2_align, loss_u_2_align, consistency_weight * consistency_loss))


        if epoch % 5 == 0:
            model1.eval()
            model2.eval()
            dice_sample = test_3d_patch.var_all_case_LA(model1, num_classes=num_classes, patch_size=patch_size,
                                                        stride_xy=18, stride_z=4)
            dice_sample2 = test_3d_patch.var_all_case_LA(model2, num_classes=num_classes, patch_size=patch_size,
                                                         stride_xy=18, stride_z=4)
            mean_dice_sample = test_3d_patch.var_all_case_LA_mean(model1, model2, num_classes=num_classes,
                                                                  patch_size=patch_size, stride_xy=18, stride_z=4)

            if dice_sample > best_dice:
                best_dice = round(dice_sample, 4)
                save_mode_path = os.path.join(self_snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                save_best_path = os.path.join(self_snapshot_path, 'best_model.pth')
                torch.save(model1.state_dict(), save_mode_path)
                torch.save(model1.state_dict(), save_best_path)
                logging.info("save best model to {}".format(save_mode_path))
                logging.info("cur dice %.4f, max dice %.4f" % (dice_sample, best_dice))

            if dice_sample2 > best_dice2:
                best_dice2 = round(dice_sample2, 4)
                save_mode_path = os.path.join(self_snapshot_path,
                                              'iter_{}_dice_{}_res.pth'.format(iter_num, best_dice2))
                save_best_path = os.path.join(self_snapshot_path, 'best_model_res.pth')
                torch.save(model2.state_dict(), save_mode_path)
                torch.save(model2.state_dict(), save_best_path)
                logging.info("resnet cur dice %.4f, max dice %.4f" % (dice_sample2, best_dice2))

            if mean_dice_sample > mean_best_dice:
                mean_best_dice = round(mean_dice_sample, 4)
                save_mode_path1 = os.path.join(self_snapshot_path,
                                               'iter_{}_dice_{}_v.pth'.format(iter_num, mean_best_dice))
                save_best_path1 = os.path.join(self_snapshot_path, 'best_model_v.pth')

                save_mode_path2 = os.path.join(self_snapshot_path,
                                               'iter_{}_dice_{}_r.pth'.format(iter_num, mean_best_dice))
                save_best_path2 = os.path.join(self_snapshot_path, 'best_model_r.pth')

                torch.save(model1.state_dict(), save_mode_path1)
                torch.save(model1.state_dict(), save_best_path1)

                torch.save(model2.state_dict(), save_mode_path2)
                torch.save(model2.state_dict(), save_best_path2)

                logging.info("mean save best model to {}".format(save_mode_path1))
                logging.info("mean cur dice %.4f, max dice %.4f" % (mean_dice_sample, mean_best_dice))

            model1.train()
            model2.train()


if __name__ == "__main__":
    ## make logger file
    pre_snapshot_path = "./model/hyperVLM/LA_{}_{}_labeled/pre_train".format(args.exp, args.labelnum)
    self_snapshot_path = "./model/hyperVLM/LA_{}_{}_labeled/self_train".format(args.exp, args.labelnum)
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        if os.path.exists(snapshot_path + '/code'):
            shutil.rmtree(snapshot_path + '/code')
    shutil.copy(__file__, self_snapshot_path)
    # -- Pre-Training
    logging.basicConfig(filename=pre_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    pre_train(args, pre_snapshot_path)
    # -- Self-training
    logging.basicConfig(filename=self_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)

import os
import argparse
import torch
import pdb
import torch.nn as nn

from utils.test_3d_patch import *

from networks.Vnet import VNet
from networks.ResVNet import ResVNet
from dataloaders.dataloader import get_network

# from testutildtc import *
# from test_usenet.dtc import VNet
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./Datasets/la', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='hyperSemi', help='exp_name')
parser.add_argument('--model', type=str, default='VNet', help='model_name')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--detail', type=int, default=1, help='print metrics for every samples?')
parser.add_argument('--nms', type=int, default=1, help='apply NMS post-processing?')
parser.add_argument('--labelnum', type=int, default=8, help='labeled data')
parser.add_argument('--stage_name', type=str, default='self_train', help='self_train or pre_train')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--n_ctx', type=int, default=4,help='prompt max len')
parser.add_argument('--dataset', type=str, default='LA',help='Name of Experiment')
parser.add_argument('--base_lr', type=float, default=1e-3, help='maximum epoch number to train')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = "./model/hyperVLM/LA_{}_{}_labeled/{}".format(FLAGS.exp, FLAGS.labelnum, FLAGS.stage_name)
test_save_path = "./model/hyperVLM/LA_{}_{}_labeled/{}_predictions/".format(FLAGS.exp, FLAGS.labelnum, FLAGS.model)
num_classes = 2

if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
print(test_save_path)
with open(FLAGS.root_path + '/data_split/test.txt', 'r') as f:
    image_list = f.readlines()
image_list = [FLAGS.root_path + "/data/" + item.replace('\n', '') + "/mri_norm2.h5" for item in
              image_list]


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


def testLA():

    model1, model2, optimizer, optimizer2 = get_network(FLAGS)

    model_path1 = os.path.join(snapshot_path, 'best_model.pth')
    model_path2 = os.path.join(
        snapshot_path,
        'best_model_resnet.pth' if FLAGS.stage_name == 'pre_train' else 'best_model_res.pth'
    )

    state1 = torch.load(str(model_path1), map_location=FLAGS.device)
    state2 = torch.load(str(model_path2), map_location=FLAGS.device)
    if FLAGS.stage_name == 'pre_train':
        state1 = state1['net']
        state2 = state2['net']

    model1.load_state_dict(state1)
    shared_prefixes = ('module.prompt_maker.', 'module.text_hyp_proj.')
    state2 = {k: v for k, v in state2.items() if not k.startswith(shared_prefixes)}
    model2.load_state_dict(state2, strict=False)

    model1.eval()
    model2.eval()

    avg_metric1 = test_all_case(model1, image_list, num_classes=num_classes,
                                patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                                save_result=False, test_save_path=test_save_path,
                                metric_detail=FLAGS.detail, nms=FLAGS.nms)

    avg_metric2 = test_all_case(model2, image_list, num_classes=num_classes,
                                patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                                save_result=False, test_save_path=test_save_path,
                                metric_detail=FLAGS.detail, nms=FLAGS.nms)

    avg_metric3 = test_all_case_average(model1, model2, image_list, num_classes=num_classes,
                                        patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                                        save_result=False, test_save_path=test_save_path,
                                        metric_detail=FLAGS.detail, nms=FLAGS.nms)

    print("v-net")
    print(avg_metric1)

    print("resvnet")
    print(avg_metric2)

    print("average")
    print(avg_metric3)




if __name__ == '__main__':
    testLA()


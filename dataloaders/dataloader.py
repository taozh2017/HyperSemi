import numpy as np
import torch
import h5py
import random

from torch import nn as nn, optim as optim
from torch.utils.data import DataLoader
from networks.Vnet import VNet
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from networks.ResVNet import ResVNet
from model.CLIP.clip import clip
from model.CoOp import PromptMaker


class HyperbolicProjectionHead(nn.Module):
    def __init__(self, in_dim=512, out_dim=256, c=1.0):
        super(HyperbolicProjectionHead, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.c = c
        self.eps = 1e-5
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def exp_map0(self, x):
        sqrt_c = self.c ** 0.5
        u_norm = torch.norm(x, dim=-1, keepdim=True)
        gamma = torch.tanh(sqrt_c * u_norm)
        denom = sqrt_c * u_norm
        factor = torch.where(denom < self.eps, torch.ones_like(denom), gamma / denom)
        return x * factor

    def log_map0(self, y):
        """ 
        Poincaré Ball -> Euclidean (Tangent space at origin).
        Formula: log_0(y) = arctanh(sqrt(c) * ||y||) * (y / (sqrt(c) * ||y||))
        """
        sqrt_c = self.c ** 0.5
        y_norm = torch.norm(y, dim=-1, keepdim=True)
        max_norm = (1.0 / sqrt_c) - 1e-5
        y_norm = torch.clamp(y_norm, min=self.eps, max=max_norm)
        val = torch.atanh(sqrt_c * y_norm)
        denom = sqrt_c * y_norm
        factor = val / denom
        factor = torch.where(denom < self.eps, torch.ones_like(denom), factor)
        return y * factor

    def project(self, x):
        sqrt_c = self.c ** 0.5
        maxnorm = (1.0 / sqrt_c) - 1e-5
        norm = torch.norm(x, dim=-1, keepdim=True)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

    def forward(self, x):
        x= x.float()
        x = self.linear(x)
        x = self.exp_map0(x)
        x = self.project(x)
        return x

class HyperbolicDistanceAttention(nn.Module):
    def __init__(self, c=0.1, temperature=1.0):
        super(HyperbolicDistanceAttention, self).__init__()
        self.c = c
        self.temperature = temperature
        self.eps = 1e-5

    def _mobius_add(self, x, y):
        """ Möbius Addition: x ⊕ y """
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        x2 = torch.sum(x ** 2, dim=-1, keepdim=True)
        y2 = torch.sum(y ** 2, dim=-1, keepdim=True)
        c = self.c
        term1 = (1 + 2 * c * xy + c * y2) * x
        term2 = (1 - c * x2) * y
        num = term1 + term2
        den = 1 + 2 * c * xy + c**2 * x2 * y2
        den = den.clamp_min(1e-15)
        return num / den

    def get_dist(self, x, y):
        sqrt_c = self.c ** 0.5
        v = self._mobius_add(-x, y)
        v_norm = torch.norm(v, p=2, dim=-1, keepdim=True)
        arg = (sqrt_c * v_norm).clamp(max=1.0 - 1e-7)
        dist = 2 * torch.atanh(arg) / sqrt_c
        return dist

    def log_map_zero(self, y):
        """ 
        Poincaré Ball -> Euclidean (Tangent space at origin).
        Formula: log_0(y) = arctanh(sqrt(c) * ||y||) * (y / (sqrt(c) * ||y||))
        """

        sqrt_c = self.c ** 0.5
        y_norm = torch.norm(y, dim=-1, keepdim=True)
        max_norm = (1.0 / sqrt_c) - 1e-5
        y_norm = torch.clamp(y_norm, min=self.eps, max=max_norm)
        val = torch.atanh(sqrt_c * y_norm)
        denom = sqrt_c * y_norm
        factor = val / denom
        factor = torch.where(denom < self.eps, torch.ones_like(denom), factor)
        return y * factor

    def exp_map_zero(self, x):
        sqrt_c = self.c ** 0.5
        u_norm = torch.norm(x, dim=-1, keepdim=True) # [B, 1]
        gamma = torch.tanh(sqrt_c * u_norm)
        denom = sqrt_c * u_norm
        factor = torch.where(denom < self.eps, torch.ones_like(denom), gamma / denom)
        return x * factor

    def forward(self, query_hyp, key_hyp, value_hyp):
        """
        Args:
            key_hyp:   [B, K, D]
            value_hyp: [B, K, D]
        """
        b = query_hyp.shape[0]
        q = query_hyp.unsqueeze(1) # [B, 1, D]
        if key_hyp.dim() == 2: k = key_hyp.unsqueeze(0).expand(b, -1, -1)
        else: k = key_hyp
        if value_hyp.dim() == 2: v = value_hyp.unsqueeze(0).expand(b, -1, -1)
        else: v = value_hyp
        dists = self.get_dist(q, k)
        scores = -dists 
        attn_weights = torch.softmax(scores / self.temperature, dim=1)
        v_tangent = self.log_map_zero(v)
        context_tangent = torch.sum(attn_weights * v_tangent, dim=1) 
        context_hyp = self.exp_map_zero(context_tangent) # [B, D]
        output_hyp = self._mobius_add(query_hyp, context_hyp)
        return output_hyp, attn_weights.squeeze(-1)

class TextVisualFusion(nn.Module):
    def __init__(self, channels=256):
        super(TextVisualFusion, self).__init__()

        self.channel_gate = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 2, channels),
            nn.Sigmoid()
        )

    def forward(self, spatial_features, global_context):
        gates = self.channel_gate(global_context)
        gates = gates.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        activated_features = spatial_features * gates
        out = spatial_features + activated_features
        
        return out

def create_Vnet(ema=False):
    net = VNet(n_channels=1, n_classes=2, normalization='instancenorm', has_dropout=True)
    net = nn.DataParallel(net)
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model
def create_ResNet(ema=False):
    net = ResVNet(n_channels=1, n_classes=2, normalization='instancenorm', has_dropout=True)
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

def get_dataset_path(dataset='pacreas', labelp='10percent'):
    files = ['train_lab.txt', 'train_unlab.txt', 'test.txt']
    return ['../Datasets/pancreas/data_split/{}'.format(f) for f in files]

# --------------------------------------------------------------------------------
# Augmentation classes
# --------------------------------------------------------------------------------
class RandomRotFlip(object):
    def __call__(self, sample):
        image, label = sample
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        return image, label

class Normalise(object):
    def __call__(self, sample):
        image, label = sample
        img_min = image.min()
        img_max = image.max()
        if img_max > img_min:
            image = (image - img_min) / (img_max - img_min)
        else:
            image = image - img_min
        return image, label

class RandomBrightness(object):
    def __init__(self, region=(0.75, 1.25)):
        self.region = region
    def __call__(self, sample):
        image, label = sample
        scale = random.uniform(self.region[0], self.region[1])
        image = image * scale
        image = np.clip(image, a_min=0., a_max=1.)
        return image, label

class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma
    def __call__(self, sample):
        image, label = sample
        noise = np.clip(self.sigma * np.random.randn(*image.shape), -2 * self.sigma, 2 * self.sigma)
        image = image + noise
        return image, label

class ToTensor(object):
    def __call__(self, sample):
        image, label = sample
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        image_t = torch.from_numpy(image)
        label_t = torch.from_numpy(label.astype(np.int64))
        return image_t, label_t

class RandomCrop(object):
    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def _get_transform(self, x):
        if x.shape[0] <= self.output_size[0] or x.shape[1] <= self.output_size[1] or x.shape[2] <= self.output_size[2]:
            pw = max((self.output_size[0] - x.shape[0]) // 2 + 1, 0)
            ph = max((self.output_size[1] - x.shape[1]) // 2 + 1, 0)
            pd = max((self.output_size[2] - x.shape[2]) // 2 + 1, 0)
            x = np.pad(x, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        else:
            pw, ph, pd = 0, 0, 0

        (w, h, d) = x.shape
        w1 = np.random.randint(0, w - self.output_size[0] + 1)
        h1 = np.random.randint(0, h - self.output_size[1] + 1)
        d1 = np.random.randint(0, d - self.output_size[2] + 1)

        def do_transform(image):
            if image.shape[0] <= self.output_size[0] or image.shape[1] <= self.output_size[1] or image.shape[2] <= self.output_size[2]:
                image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return image

        return do_transform

    def __call__(self, samples):
        transform = self._get_transform(samples[0])
        return [transform(s) for s in samples]

class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def _get_transform(self, label):
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 1, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 1, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 1, 0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        else:
            pw, ph, pd = 0, 0, 0

        (w, h, d) = label.shape
        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        def do_transform(x):
            if x.shape[0] <= self.output_size[0] or x.shape[1] <= self.output_size[1] or x.shape[2] <= self.output_size[2]:
                x = np.pad(x, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            x = x[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return x

        return do_transform

    def __call__(self, samples):
        transform = self._get_transform(samples[0])
        return [transform(s) for s in samples]

# Update Pancreas transform to include full augmentation pipeline
class Pancreas(Dataset):
    def __init__(self, base_dir, name, split, no_crop=False, labelp=20, reverse=False, TTA=False):
        self._base_dir = base_dir
        self.split = split
        self.reverse = reverse
        self.labelp = '10percent'
        if labelp == 20:
            self.labelp = '20percent'

        tr_transform = Compose([
            #RandomRotFlip(),
            #Normalise(),
            #RandomBrightness((0.75, 1.25)),
            #RandomNoise(mu=0, sigma=0.1),
            RandomCrop((96, 96, 96)),
            ToTensor()
        ])
        if no_crop:
            test_transform = Compose([
                CenterCrop((96, 96, 96)),
                ToTensor()
            ])
        else:
            test_transform = Compose([
                CenterCrop((96, 96, 96)),
                ToTensor()
            ])

        data_list_paths = get_dataset_path(name, self.labelp)

        if split == 'train_lab':
            data_path = data_list_paths[0]
            self.transform = tr_transform
        elif split == 'train_unlab':
            data_path = data_list_paths[1]
            self.transform = tr_transform
        else:
            data_path = data_list_paths[2]
            self.transform = test_transform

        with open(data_path, 'r') as f:
            self.image_list = f.readlines()

        self.image_list = [self._base_dir + "/{}".format(item.strip()) for item in self.image_list]
        print("Split : {}, total {} samples".format(split, len(self.image_list)))

    def __len__(self):
        if self.split == 'train_lab' and self.labelp == '20percent':
            return len(self.image_list) * 5
        elif self.split == 'train_lab' and self.labelp == '10percent':
            return len(self.image_list) * 10
        else:
            return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx % len(self.image_list)]
        if self.reverse:
            image_path = self.image_list[len(self.image_list) - idx % len(self.image_list) - 1]
        h5f = h5py.File(image_path+'.h5', 'r')
        image, label = h5f['image'][:], h5f['label'][:].astype(np.float32)
        samples = (image, label)
        if self.transform:
            image_, label_ = self.transform(samples)
        return image_.float(), label_.long()

class VL_Network_Wrapper(nn.Module):
    def __init__(self, vision_net, prompt_maker, text_hyp_proj, args):
        super().__init__()
        self.vision_net = vision_net
        self.prompt_maker = prompt_maker
        self.args = args
        self.hyp_dim = 256    
        self.clip_dim = 512   
        self.vision_dim = 256
        self.curvature = 0.1  
        self.vision_hyp_proj = HyperbolicProjectionHead(in_dim=self.vision_dim, out_dim=self.hyp_dim, c=self.curvature)
        self.text_hyp_proj = text_hyp_proj
        self.cross_attn = HyperbolicDistanceAttention(c=self.curvature) 
        self.fusion_module = TextVisualFusion(channels=self.vision_dim)

    def forward(self, x):
        text_features = self.prompt_maker() 

        if text_features.dim() == 2:
            raw_text_feat = text_features # [K, 512]
        else:
            raw_text_feat = text_features[0]
        
        t_hyp = self.text_hyp_proj(raw_text_feat) 
        alignment_tools = {
            't_hyp': t_hyp,
            'vision_hyp_proj': self.vision_hyp_proj,
            'cross_attn': self.cross_attn,
            'fusion_module': self.fusion_module
        }
        out = self.vision_net(x, alignment_tools)
        return out


def get_parameter_groups(network, base_lr, prompt_lr_scale=50):

    if isinstance(network, nn.DataParallel):
        network = network.module
        
    prompt_params = []
    base_params = []
    
    for name, param in network.named_parameters():
        if not param.requires_grad:
            continue
            
        if "prompt_learner" in name:
            prompt_params.append(param)
        else:
            base_params.append(param)
            
    print(f"  - Prompt Params: {len(prompt_params)} (LR: {base_lr * prompt_lr_scale})")
    print(f"  - Base Params:   {len(base_params)} (LR: {base_lr})")
    
    return [
        {'params': base_params, 'lr': base_lr},
        {'params': prompt_params, 'lr': base_lr * prompt_lr_scale}
    ]

def get_network(args=None):
    
    clip_model, _ = clip.load("ViT-B/32", device=args.device)
    for p in clip_model.parameters():
        p.requires_grad = False
    if "Pancreas" in args.dataset:
        prompts = ["background", "pancreas"]
    elif "LA" in args.dataset:
        prompts = ["background", "left atrium"]
    else:
        prompts = ["background", "target"]

    prompt_maker_1 = PromptMaker(args=args, prompts=prompts, clip_model=clip_model, n_ctx=args.n_ctx).to(args.device)
    text_hyp_proj = HyperbolicProjectionHead(in_dim=512, out_dim=256, c=0.1).to(args.device)
    
    raw_vnet = VNet(n_channels=1, n_classes=2, n_filters=16, normalization='instancenorm', has_dropout=True)
    
    net1 = VL_Network_Wrapper(raw_vnet, prompt_maker_1, text_hyp_proj, args).to(args.device)
    net1 = nn.DataParallel(net1)

    raw_resnet = ResVNet(n_channels=1, n_classes=2, n_filters=16, normalization='instancenorm', has_dropout=True)
    
    net2 = VL_Network_Wrapper(raw_resnet, prompt_maker_1, text_hyp_proj, args).to(args.device)
    net2 = nn.DataParallel(net2)

    prompt_lr_scale = 1 
    
    params_group1 = get_parameter_groups(net1, base_lr=args.base_lr, prompt_lr_scale=prompt_lr_scale)
    optimizer1 = optim.Adam(params_group1, weight_decay=1e-4)

    params_group2 = get_parameter_groups(net2, base_lr=args.base_lr, prompt_lr_scale=prompt_lr_scale)
    optimizer2 = optim.Adam(params_group2, weight_decay=1e-4)


    return net1, net2, optimizer1, optimizer2

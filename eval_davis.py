import os
from os import path
import time
from argparse import ArgumentParser

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from model.eval_network import STCN
from dataset.davis_test_dataset import DAVISTestDataset
from util.tensor_util import unpad
from inference_core import InferenceCore

from progressbar import progressbar


"""
Arguments loading
"""
parser = ArgumentParser()
parser.add_argument('--model', default='saves/stcn.pth')
parser.add_argument('--davis_path', default='../DAVIS/2017')
parser.add_argument('--output',default='davis_output/tmp/')
parser.add_argument('--split', help='val/testdev', default='val')
parser.add_argument('--top', type=int, default=20)
parser.add_argument('--amp', action='store_true')
parser.add_argument('--mem_every', default=5, type=int)
parser.add_argument('--include_last', help='include last frame as temporary memory?', action='store_true')
parser.add_argument('--affinity',choices=['L2','Cosine'],default='L2')
parser.add_argument('--thresh_close',type=int,default=1)
parser.add_argument('--memory_budget',type=int,default=5001)
args = parser.parse_args()

davis_path = args.davis_path
out_path = args.output

# Simple setup
os.makedirs(out_path, exist_ok=True)
palette = Image.open(path.expanduser(davis_path + '/trainval/Annotations/480p/blackswan/00000.png')).getpalette()

torch.autograd.set_grad_enabled(False)

# Setup Dataset
if args.split == 'val':
    test_dataset = DAVISTestDataset(davis_path+'/trainval', imset='2017/val.txt')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
elif args.split == 'testdev':
    test_dataset = DAVISTestDataset(davis_path+'/test-dev', imset='2017/test-dev.txt')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
else:
    raise NotImplementedError

# Load our checkpoint
src_dict = torch.load(args.model)
top_k = args.top
prop_model = STCN().cuda().eval()
try:
    prop_model.load_state_dict(src_dict)
except:
    print('Seems to be stage 0 results')
    # Maps SO weight (without other_mask) to MO weight (with other_mask)
    for k in list(src_dict.keys()):
        if k == 'value_encoder.conv1.weight':
            if src_dict[k].shape[1] == 4:
                pads = torch.zeros((64,1,7,7), device=src_dict[k].device)
                nn.init.orthogonal_(pads)
                src_dict[k] = torch.cat([src_dict[k], pads], 1)
    prop_model.load_state_dict(src_dict)


total_process_time = 0
total_frames = 0

# Start eval
for data in progressbar(test_loader, max_value=len(test_loader), redirect_stdout=True):

    with torch.cuda.amp.autocast(enabled=args.amp):
        rgb = data['rgb'].cuda()
        msk = data['gt'][0].cuda()
        info = data['info']
        name = info['name'][0]
        k = len(info['labels'][0])
        size = info['size_480p']

        torch.cuda.synchronize()
        process_begin = time.time()

        processor = InferenceCore(prop_model, rgb, k, top_k=top_k, 
                        mem_every=args.mem_every, include_last=args.include_last,
                        memory_budget=args.memory_budget,close_thresh=args.thresh_close)
        processor.interact(msk[:,0], 0, rgb.shape[1])

        # Do unpad -> upsample to original size 
        out_masks = torch.zeros((processor.t, 1, *size), dtype=torch.uint8, device='cuda')
        for ti in range(processor.t):
            prob = unpad(processor.prob[:,ti], processor.pad)
            prob = F.interpolate(prob, size, mode='bilinear', align_corners=False)
            out_masks[ti] = torch.argmax(prob, dim=0)
        
        out_masks = (out_masks.detach().cpu().numpy()[:,0]).astype(np.uint8)

        torch.cuda.synchronize()
        total_process_time += time.time() - process_begin
        total_frames += out_masks.shape[0]

        

        # Save keys
        # os.makedirs(path.join(out_path, 'keys'),exist_ok=True)
        # this_key_path = path.join(out_path,'keys', name)
        # keys = processor.mem_bank.mem_k.view(1,64,-1,processor.kh,processor.kw)
        # np.save(this_key_path,keys.cpu().numpy())

        #Save indices and values
        # os.makedirs(path.join(out_path, 'indices'),exist_ok=True)
        # this_ind_path = path.join(out_path, 'indices', name)
        # ind = processor.mem_bank.indices
        # np.save(this_ind_path, ind.cpu().numpy())

        # os.makedirs(path.join(out_path, 'values'), exist_ok=True)
        # this_val_path = path.join(out_path, 'values', name)
        # values = processor.mem_bank.values
        # np.save(this_val_path, values.cpu().numpy())


        # Save the results
        this_out_path = path.join(out_path, name)
        os.makedirs(this_out_path, exist_ok=True)

        for f in range(out_masks.shape[0]):
            img_E = Image.fromarray(out_masks[f])
            img_E.putpalette(palette)
            img_E.save(os.path.join(this_out_path, '{:05d}.png'.format(f)))

        del rgb
        del msk
        del processor

print('Total processing time: ', total_process_time)
print('Total processed frames: ', total_frames)
print('FPS: ', total_frames / total_process_time)
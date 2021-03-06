"""
network.py - The core of the neural network
Defines the structure and memory operations
Modifed from STM: https://github.com/seoungwugoh/STM

The trailing number of a variable usually denote the stride
e.g. f16 -> encoded features with stride 16
"""

import math

import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F

from model.modules import *


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.compress = ResBlock(1024, 512)
        self.up_16_8 = UpsampleBlock(512, 512, 256) # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(256, 256, 256) # 1/8 -> 1/4

        self.pred = nn.Conv2d(256, 1, kernel_size=(3,3), padding=(1,1), stride=1)

    def forward(self, f16, f8, f4):
        x = self.compress(f16)
        x = self.up_16_8(f8, x)
        x = self.up_8_4(f4, x)

        x = self.pred(F.relu(x))
        
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        return x

class MyDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super(MyDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p

    def forward(self, X):
        binomial = torch.distributions.binomial.Binomial(probs=1-self.p)
        return X * binomial.sample(X.size()).to(device=X.get_device()) * (1.0/(1-self.p))

class MemoryReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = MyDropout()
        
 
    def get_affinity(self, mk, qk,method='L2'):
        B, CK, T, H, W = mk.shape
        if method=='L2':
            mk = mk.flatten(start_dim=2)
            qk = qk.flatten(start_dim=2)

            a = mk.pow(2).sum(1).unsqueeze(2)
            b = 2 * (mk.transpose(1, 2) @ qk)
            # We don't actually need this, will update paper later
            # c = qk.pow(2).sum(1).unsqueeze(1)
            affinity = (-a+b) / math.sqrt(CK)   # B, THW, HW

        elif method=='Cosine':
            mk = mk.flatten(start_dim=2) #B, CK, THW
            qk = qk.flatten(start_dim=2) #B, CK, HW
            affinity = mk.transpose(1,2) @ qk
        else:
            raise NotImplementedError

        return affinity

    def readout(self, affinity, mv, qv, mask=None):
        B, CV, T, H, W = mv.shape

        affinity = self.reallocate(affinity, mask=mask)
        # affinity = self.dropout(affinity)
        affinity_softmax = F.softmax(affinity, dim=1)

        mo = mv.view(B, CV, T*H*W)
        mem = torch.bmm(mo, affinity_softmax) # Weighted-sum B, CV, HW
        mem = mem.view(B, CV, H, W)

        mem_out = torch.cat([mem, qv], dim=1)

        return mem_out

    def reallocate(self, affinity, p=0.5, mask=None):
        B, THW, HW = affinity.shape

        mask = mask.view(B,-1)

        support_attention = affinity.mean(dim=2, keepdim=False)  # B, THW
        masked_support = (mask > 0.5) * support_attention

        sorted_idx = torch.argsort(masked_support,dim=1,descending=True)
        obj_cnt = torch.count_nonzero(masked_support,dim=1)

        w = nn.Parameter(torch.arange(1,
                                      obj_cnt.max()+1,
                                      dtype=torch.float),
                         requires_grad=False).to(
            device=affinity.get_device()).half()

        new_affinity = affinity.clone()

        for i in range(B):
            k = obj_cnt[i]
            if k == 0:
                continue
            idx = torch.randint(high=k.item(), size=(int(k*p),))
            w_i = w[idx] / (sum(w[idx])/k)

            new_affinity[i,sorted_idx[i,idx],:] = affinity[i,sorted_idx[i,idx],:] * w_i.unsqueeze(1) # B, THW

        return new_affinity

    # def reallocate(self, affinity, p=0.5, masks=None):
    #     B, THW, HW = affinity.shape
    #     T = THW/HW
    #     support_attention = affinity.mean(dim=2,keepdim=True) # B, THW, 1
    #     k = int(THW*p)
    #     topk_v, topk_i = torch.topk(support_attention,k=k,dim=1) #1,k
    #     w = nn.Parameter(torch.arange(1,k+1,dtype=torch.float),requires_grad=False).to(device=affinity.get_device()).half()
    #     w = w/ (sum(w)/k)
    #     w = w.view(1,k,1).repeat(B,1,1)
    #     support_attention = support_attention.scatter(dim=1,index=topk_i,src=topk_v*w) #B, THW, 1
    #     new_affinity = support_attention * affinity
    #     return new_affinity

class STCN(nn.Module):
    def __init__(self, single_object):
        super().__init__()
        self.single_object = single_object

        self.key_encoder = KeyEncoder()
        if single_object:
            self.value_encoder = ValueEncoderSO() 
        else:
            self.value_encoder = ValueEncoder() 

        # Projection from f16 feature space to key space
        self.key_proj = KeyProjection(1024, keydim=64)

        # Compress f16 a bit to use in decoding later on
        self.key_comp = nn.Conv2d(1024, 512, kernel_size=3, padding=1)

        self.memory = MemoryReader()
        self.decoder = Decoder()

    def aggregate(self, prob):
        new_prob = torch.cat([
            torch.prod(1-prob, dim=1, keepdim=True),
            prob
        ], 1).clamp(1e-7, 1-1e-7)
        logits = torch.log((new_prob /(1-new_prob)))
        return logits

    def encode_key(self, frame): 
        # input: b*t*c*h*w
        b, t = frame.shape[:2]

        f16, f8, f4 = self.key_encoder(frame.flatten(start_dim=0, end_dim=1))
        k16 = self.key_proj(f16)
        f16_thin = self.key_comp(f16)

        # B*C*T*H*W
        k16 = k16.view(b, t, *k16.shape[-3:]).transpose(1, 2).contiguous()

        # B*T*C*H*W
        f16_thin = f16_thin.view(b, t, *f16_thin.shape[-3:])
        f16 = f16.view(b, t, *f16.shape[-3:])
        f8 = f8.view(b, t, *f8.shape[-3:])
        f4 = f4.view(b, t, *f4.shape[-3:])

        return k16, f16_thin, f16, f8, f4

    def encode_value(self, frame, kf16, mask, other_mask=None): 
        # Extract memory key/value for a frame
        if self.single_object:
            f16 = self.value_encoder(frame, kf16, mask)
        else:
            f16 = self.value_encoder(frame, kf16, mask, other_mask)
        return f16.unsqueeze(2) # B*512*T*H*W

    def segment(self, qk16, qv16, qf8, qf4, mk16, mv16, selector=None, affinity='L2', masks=None, sec_masks=None):
        # q - query, m - memory
        # qv16 is f16_thin above

        _,_,fh,fw = qk16.shape
        B,N,C,H,W = masks.shape


        masks = masks.view(-1,C,H,W)
        masks = F.interpolate(masks,
                              size=(fh, fw),
                              mode='bilinear',
                              align_corners=True)

        masks = masks.view(B,N,C,fh,fw)


        affinity = self.memory.get_affinity(mk16,
                                            qk16,
                                            method=affinity)

        if self.single_object:
            logits = self.decoder(self.memory.readout(affinity, mv16, qv16, mask=masks)
                                  , qf8, qf4)
            prob = torch.sigmoid(logits)
        else:
            sec_masks = sec_masks.view(-1, C, H, W)
            sec_masks = F.interpolate(sec_masks, size=(fh, fw), mode='bilinear')
            sec_masks = sec_masks.view(B,N, C, fh, fw)

            logits = torch.cat([
                self.decoder(self.memory.readout(affinity, mv16[:,0],
                                                 qv16, mask=masks),
                             qf8, qf4),
                self.decoder(self.memory.readout(affinity, mv16[:,1],
                                                 qv16, mask=sec_masks)
                             , qf8, qf4),
            ], 1)

            prob = torch.sigmoid(logits)
            prob = prob * selector.unsqueeze(2).unsqueeze(2)

        logits = self.aggregate(prob)
        prob = F.softmax(logits, dim=1)[:, 1:]

        return logits, prob

    def forward(self, mode, *args, **kwargs):
        if mode == 'encode_key':
            return self.encode_key(*args, **kwargs)
        elif mode == 'encode_value':
            return self.encode_value(*args, **kwargs)
        elif mode == 'segment':
            return self.segment(*args, **kwargs)
        else:
            raise NotImplementedError



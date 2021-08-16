import numpy as np
import torch
import torch.nn.functional as NF
import math
from torch_scatter import scatter_mean

def softmax_w_top(x, top):
    values, indices = torch.topk(x, k=top, dim=1)
    val = values.clone()
    x_exp = values.exp_()

    x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
    x.zero_().scatter_(1, indices, x_exp) # B * THW * HW

    return x , indices, val

class FeatureBank:

    def __init__(self, obj_n, memory_budget, device, update_rate=0.1, thres_close=0.95, top_k=20):
        self.num_objects = obj_n
        self.top_k = top_k

        self.CK = None
        self.CV = None

        self.mem_k = None
        self.mem_v = None


        self.update_rate = update_rate
        self.thres_close = thres_close
        self.device = device

        # self.info = [None for _ in range(obj_n)]
        self.peak_n = np.zeros(1)
        self.replace_n = np.zeros(1)

        self.affinity = None

        self.budget = memory_budget
        # if obj_n == 2:
        #     self.class_budget = 0.8 * self.class_budget

    def _global_matching(self, mk, qk):
        # NE means number of elements -- typically T*H*W
        B, CK, NE = mk.shape

        a = mk.pow(2).sum(1).unsqueeze(2)
        b = 2 * (mk.transpose(1, 2) @ qk)
        # We don't actually need this, will update paper later
        c = qk.pow(2).expand(B, -1, -1).sum(1).unsqueeze(1)

        affinity = (-a+b-c) / math.sqrt(CK)  # B, NE, HW

        self.affinity = affinity.clone()

        affinity, indices, values = softmax_w_top(affinity, top=self.top_k)  # B, THW, HW


        val, cnt = torch.unique(indices,return_counts= True)

        self.info[val,1] += torch.log(cnt)

        return affinity

    def _readout(self, affinity, mv):
        return torch.bmm(mv, affinity)

    def match_memory(self, qk):
        k = self.num_objects
        _, _, h, w = qk.shape

        qk = qk.flatten(start_dim=2)


        mk = self.mem_k
        mv = self.mem_v

        affinity = self._global_matching(mk, qk)


        # One affinity for all
        readout_mem = self._readout(affinity.expand(k, -1, -1), mv)

        return readout_mem.view(k, self.CV, h, w)

    def add_memory(self, key, value, frame_idx=1):
        # Temp is for "last frame"
        # Not always used
        # But can always be flushed
        self.temp_k = None
        self.temp_v = None
        key = key.flatten(start_dim=2)
        value = value.flatten(start_dim=2)

        if self.mem_k is None:
            # First frame, just shove it in
            self.mem_k = key
            self.mem_v = value
            self.CK = key.shape[1]
            self.CV = value.shape[1]

            self.info = torch.zeros((self.mem_k.shape[-1], 2), device=self.device)
            self.info[:, 0] = frame_idx
            self.peak_n = max(self.peak_n, self.info.shape[0])
        else:
            # self.mem_k = torch.cat([self.mem_k, key], 2)
            # self.mem_v = torch.cat([self.mem_v, value], 2)
            self.update(key,value,frame_idx)

    def update(self,key,value,frame_idx):
        corr = self.affinity[0]

        _, Ck, bank_n = self.mem_k.shape
        Cv = self.mem_v.shape[1]


        related_bank_idx = corr.argmax(dim=0, keepdim=True) #1,HW
        related_bank_corr = -torch.gather(corr, 0, related_bank_idx) #1,HW

        # print(related_bank_corr.min(), related_bank_corr.max())

        related_bank_corr = 1/(torch.exp(related_bank_corr))

        # print(self.thres_close)
        # print(related_bank_corr.min(),related_bank_corr.max())




        # greater than threshold, merge them
        selected_idx = (related_bank_corr[0] > self.thres_close).nonzero()

        # print(f'merg% : {(related_bank_corr[0] > self.thres_close).sum()/related_bank_corr.shape[1]}')

        class_related_bank_idx = related_bank_idx[0, selected_idx[:, 0]]  # selected_HW
        unique_related_bank_idx, cnt = class_related_bank_idx.unique(dim=0, return_counts=True)

        # Update key
        key_bank_update = torch.zeros((Ck, bank_n), dtype=torch.float, device=self.device)  # d_key, THW
        key_bank_idx = class_related_bank_idx.unsqueeze(0).expand(Ck, -1)  # d_key, HW
        scatter_mean(self.mem_k[0][:, selected_idx[:, 0]], key_bank_idx, dim=1, out=key_bank_update)

        self.mem_k[0][:, unique_related_bank_idx] = \
            ((1 - self.update_rate) * self.mem_k[0][:, unique_related_bank_idx] + \
             self.update_rate * key_bank_update[:, unique_related_bank_idx])

        for class_idx in range(self.num_objects):
            # Update value
            val_bank_update = torch.zeros((Cv, bank_n), dtype=torch.float, device=self.device)
            val_bank_idx = class_related_bank_idx.unsqueeze(0).expand(Cv, -1)
            scatter_mean(self.mem_v[class_idx][:, selected_idx[:, 0]], val_bank_idx, dim=1, out=val_bank_update)

            self.mem_v[class_idx][:, unique_related_bank_idx] = \
                ((1 - self.update_rate) * self.mem_v[class_idx][:, unique_related_bank_idx] + \
                 self.update_rate * val_bank_update[:, unique_related_bank_idx])

        # less than the threshold, concat them
        selected_idx = (related_bank_corr[0] <= self.thres_close).nonzero()

        if self.budget < bank_n + selected_idx.shape[0]:
            self.remove(selected_idx.shape[0], frame_idx)

        self.mem_k = torch.cat([self.mem_k, key[:,:, selected_idx[:, 0]]], dim=2)
        # for class_idx in range(self.num_objects):
        self.mem_v = torch.cat([self.mem_v, value[:, :, selected_idx[:, 0]]], dim=2)

        new_info = torch.zeros((selected_idx.shape[0], 2), device=self.device)
        new_info[:, 0] = frame_idx
        self.info = torch.cat([self.info, new_info], dim=0)

        self.peak_n = max(self.peak_n, self.info.shape[0])

        self.info[:, 1] = torch.clamp(self.info[:, 1], 0, 1e5)


    def remove(self, request_n, frame_idx):

        old_size = self.mem_k.shape[-1]

        LFU = frame_idx - self.info[:, 0]  # time length
        LFU = self.info[:, 1] / LFU
        thres_dynamic = int(LFU.min()) + 1
        iter_cnt = 0

        while True:
            selected_idx = LFU > thres_dynamic
            self.mem_k = self.mem_k[:,:, selected_idx]
            # for class_idx in range(self.num_objects):
            self.mem_v = self.mem_v[:,:, selected_idx]
            self.info = self.info[selected_idx]
            LFU = LFU[selected_idx]
            iter_cnt += 1

            balance = (self.budget - self.mem_k.shape[-1]) - request_n
            if balance < 0:
                thres_dynamic = int(LFU.min()) + 1
            else:
                break

        new_size = self.mem_k.shape[-1]
        self.replace_n += old_size - new_size

        return balance

    def print_peak_mem(self):

        ur = self.peak_n / self.class_budget
        rr = self.replace_n / self.class_budget
        print(f'Obj num: {self.obj_n}.', f'Budget / obj: {self.class_budget}.', f'UR: {ur}.', f'Replace: {rr}.')
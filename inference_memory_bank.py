import math
import torch


def softmax_w_top(x, top):
    values, indices = torch.topk(x, k=top, dim=1)
    val = values.clone()
    x_exp = values.exp_()

    x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
    x.zero_().scatter_(1, indices, x_exp) # B * THW * HW

    return x , indices, val


class MemoryBank:
    def __init__(self, k, top_k=20):
        self.top_k = top_k

        self.CK = None
        self.CV = None

        self.mem_k = None
        self.mem_v = None

        self.num_objects = k

        self.indices = None
        self.values = None




    def _global_matching(self, mk, qk):
        # NE means number of elements -- typically T*H*W
        B, CK, NE = mk.shape

        a = mk.pow(2).sum(1).unsqueeze(2)
        b = 2 * (mk.transpose(1, 2) @ qk)
        # We don't actually need this, will update paper later
        # c = qk.pow(2).expand(B, -1, -1).sum(1).unsqueeze(1)

        affinity = (-a+b) / math.sqrt(CK)  # B, NE, HW
        affinity, indices, values = softmax_w_top(affinity, top=self.top_k)  # B, THW, HW
        if self.indices is None:
            self.indices = indices
            self.values = values
        else:
            self.indices = torch.cat([self.indices,indices],dim=0)
            self.values = torch.cat([self.values,values],dim=0)
        return affinity

    def _readout(self, affinity, mv):
        return torch.bmm(mv, affinity)

    def match_memory(self, qk):
        k = self.num_objects
        _, _, h, w = qk.shape

        qk = qk.flatten(start_dim=2)
        
        if self.temp_k is not None:
            mk = torch.cat([self.mem_k, self.temp_k], 2)
            mv = torch.cat([self.mem_v, self.temp_v], 2)
        else:
            mk = self.mem_k
            mv = self.mem_v

        affinity = self._global_matching(mk, qk)

        # One affinity for all
        readout_mem = self._readout(affinity.expand(k,-1,-1), mv)

        return readout_mem.view(k, self.CV, h, w)

    def add_memory(self, key, value, is_temp=False):
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
        else:
            if is_temp:
                self.temp_k = key
                self.temp_v = value
            else:
                self.mem_k = torch.cat([self.mem_k, key], 2)
                self.mem_v = torch.cat([self.mem_v, value], 2)

        # print(self.mem_k.shape,self.mem_v.shape)
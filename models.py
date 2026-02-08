import torch

class BlockWrapper(torch.nn.Module):
    def __init__(self, block, hidden_dim, vec=None):
        super().__init__()
        self.multiplier = 1.0
        self.block = block

        try:
            ref_param = next(block.parameters())
            init_dtype = ref_param.dtype
        except StopIteration:
            init_dtype = torch.float32
            
        if vec is not None:
            self.vec = vec
        else:
            self.vec = torch.nn.Parameter(torch.zeros(hidden_dim, dtype=init_dtype))

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        if isinstance(output, tuple):
            modified_hidden = output[0] + (self.multiplier * self.vec)
            return (modified_hidden,) + output[1:]
        elif isinstance(output, torch.Tensor):
            return output + (self.multiplier * self.vec)
        else:
            return output

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.block, name)
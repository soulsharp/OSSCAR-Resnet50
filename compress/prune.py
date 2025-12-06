import torch.nn as nn


class PrunedResnet50(nn.Module):
    def __init__(self, pruned_modules_list, keep_mask_list):
        super().__init__()
        assert len(pruned_modules_list) - 1 == len(keep_mask_list)
        self.module_list = nn.ModuleList(pruned_modules_list)

        # Register masks as buffers so that they can be saved/loaded along with the model
        for i, mask in enumerate(keep_mask_list):
            self.register_buffer(f"input_mask_{i}", mask)

        self.keep_mask_list = keep_mask_list

    def forward(self, x):
        # The first module has no input mask
        x = self.module_list[0](x)

        # The mask of the ith module in module_list is the i-1th entry of keep_mask_list
        for i in range(1, len(self.module_list)):
            x = x[:, self.keep_mask_list[i - 1], :, :]
            x = self.module_list[i](x)

        return x

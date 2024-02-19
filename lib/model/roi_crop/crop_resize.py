import torch
from torch.autograd import Function
import torch.nn.functional as F

class RoICropFunction(Function):
    @staticmethod
    def forward(ctx, input1, input2):
        ctx.save_for_backward(input1, input2)
        ctx.device = input1.device
        output = torch.zeros(input2.size(0), input1.size(1), input2.size(1), input2.size(2), device=ctx.device)
        if input1.is_cuda:
            output = output.cuda(ctx.device)
        for i in range(input2.size(0)):
            output[i] = F.grid_sample(input1[i].unsqueeze(0).unsqueeze(0), input2[i].unsqueeze(0), mode='bilinear', padding_mode='zeros')
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors
        grad_input1 = torch.zeros_like(input1)
        grad_input2 = torch.zeros_like(input2)

        for i in range(input2.size(0)):
            grad_input1[i] = F.grid_sample(grad_output[i].unsqueeze(0).unsqueeze(0), input2[i].unsqueeze(0), mode='bilinear', padding_mode='zeros')
            grad_input2[i] = F.grid_sample(grad_output[i].unsqueeze(0).unsqueeze(0), input2[i].unsqueeze(0), mode='bilinear', padding_mode='zeros', align_corners=False)

        return grad_input1, grad_input2

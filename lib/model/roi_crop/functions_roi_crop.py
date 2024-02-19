import torch
import torch.nn.functional as F
from torch.autograd import Function

class RoICropFunction(Function):
    def forward(self, input1, input2):
        self.input1 = input1.clone()
        self.input2 = input2.clone()

        grid = input2.view(input2.size(0), 2, -1)
        grid = grid.transpose(1, 2).view(-1, 2, input2.size(1), input2.size(2))

        output = F.grid_sample(input1, grid)
        return output

    def backward(self, grad_output):
        grad_input1 = F.grid_sample(grad_output, -self.input2)

        # Calculate grad_input2 (this is just an example, you might need to modify based on your specific use case)
        grad_input2 = None
        if self.needs_input_grad[1]:
            grad_input2 = torch.zeros_like(self.input2)
            # You need to implement the calculation of grad_input2 based on your specific requirements

        return grad_input1, grad_input2

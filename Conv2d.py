import torch
import torch.nn as nn

class CustomConv2D(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(CustomConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # Initialize the kernel weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
    
    def forward(self, x):
        
        def apply_padding(x,pad):
            
            batch,filter,height,width = x.shape
            new_height,new_width = height + 2 * pad,width + 2 * pad
    
            padding_tensor = torch.zeros(batch,filter,new_height,new_width)
    
            padding_tensor[:,:,pad : pad + height,pad : pad + width ] = x
    
            return padding_tensor

        # Get the dimensions of the input
        N, C, H, W = x.size()
        
        # Calculate the dimensions of the output
        KH, KW = self.kernel_size, self.kernel_size
        SH, SW = self.stride, self.stride

        OH = (H - KH + 2 * self.padding) // SH + 1
        OW = (W - KW + 2 * self.padding) // SW + 1
        # Initialize the output tensor
        output = torch.zeros(N, self.out_channels, OH, OW).to(x.device)
        
        if self.padding > 0:
            x = apply_padding(x,self.padding)
        
        for i in range(OH):
            for j in range(OW):
                # Extract region from input
                region = x[:, :, i*SH:i*SH+KH, j*SW:j*SW+KW]
                
                # Compute convolution
                output[:, :, i, j] = torch.sum(region.unsqueeze(1) * self.weight, dim=[2, 3, 4]) + self.bias
        
        return output
#Custom Application and Pytorch Application Comparison   
if __name__ == "__main__":
    in_channels = 1
    out_channels = 1
    kernel_size = 3
    stride = 1
    padding = 0

    # Create input tensor
    input_tensor = torch.randn(1, in_channels, 6, 6)  # Batch size 1, 1 channel, 6x6 image

    # PyTorch Conv2d
    pytorch_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    pytorch_output = pytorch_conv(input_tensor)

    # Custom Conv2d
    custom_conv = CustomConv2D(in_channels, out_channels, kernel_size, stride, padding)
    custom_conv.weight.data = pytorch_conv.weight.data.clone()
    custom_conv.bias.data = pytorch_conv.bias.data.clone()
    custom_output = custom_conv(input_tensor)

    print("PyTorch Conv2d Output:")
    print(pytorch_output)
    print("Custom Conv2d Output:")
    print(custom_output)
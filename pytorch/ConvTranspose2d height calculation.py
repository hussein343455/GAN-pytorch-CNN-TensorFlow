nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=0) 

H_in = 5
stride = 2
kernel_size=3
padding=0
output_padding=0
dilation=1

# By default, stride=1, padding=0, and output_padding=0.
H_out= (H_in -1)*stride-2*padding+dilation*(kernel_size-1)+output_padding+1
# H_out = (5-1)*2 - 2*0 + 1*(3-1) + 0 + 1 = 11

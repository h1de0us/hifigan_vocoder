def get_padding(kernel_size, dilation=1):
        return (kernel_size * dilation - dilation) // 2
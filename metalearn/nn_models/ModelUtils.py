def conv1d_out_length(L_in: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1) -> int:
    return ((L_in + 2*padding - dilation*(kernel_size - 1) - 1) // stride) + 1

def pool1d_out_length(L_in: int, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1) -> int:
    if stride is None:
        stride = kernel_size
    return ((L_in + 2*padding - dilation*(kernel_size - 1) - 1) // stride) + 1


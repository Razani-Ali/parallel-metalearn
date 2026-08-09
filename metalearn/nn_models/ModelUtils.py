def conv1d_out_length(L_in: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1) -> int:
    """
    Calculates the spatial output length of a 1D Convolutional layer.

    Args:
        L_in (int): Input signal sequence length.
        kernel_size (int): Convolution kernel filter length.
        stride (int): Stride factor of the convolution operation. Defaults to 1.
        padding (int): Zero-padding applied to both sides of input. Defaults to 0.
        dilation (int): Spacing between kernel elements. Defaults to 1.

    Returns:
        int: Calculated output signal length after 1D convolution.
    """
    # Apply standard 1D CNN arithmetic output dimension formula
    return ((L_in + 2*padding - dilation*(kernel_size - 1) - 1) // stride) + 1

def pool1d_out_length(L_in: int, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1) -> int:
    """
    Calculates the spatial output length of a 1D Pooling layer (AvgPool1d / MaxPool1d).

    Args:
        L_in (int): Input signal sequence length.
        kernel_size (int): Pooling window size.
        stride (int, optional): Pooling stride factor. If None, defaults to kernel_size.
        padding (int): Zero-padding applied to both sides. Defaults to 0.
        dilation (int): Dilation factor of pooling window. Defaults to 1.

    Returns:
        int: Calculated output signal length after 1D pooling.
    """
    # Default stride to kernel_size if no explicit stride value is provided
    if stride is None:
        stride = kernel_size
    # Apply standard 1D pooling output dimension formula
    return ((L_in + 2*padding - dilation*(kernel_size - 1) - 1) // stride) + 1
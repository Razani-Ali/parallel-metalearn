import torch
import torch.nn as nn
from typing import Tuple
from abc import ABC, abstractmethod



class BaseScaler(nn.Module, ABC):
    """
    Abstract Base Class for all PyTorch-native feature scalers.

    Enforces a unified interface for computing statistics (`fit`) and 
    applying transformations (`forward`) on multi-dimensional tensors.
    Subclasses automatically inherit PyTorch `nn.Module` buffer management 
    and state handling capabilities.
    """

    def __init__(self, eps: float = 1e-7):
        """
        Args:
            eps (float): Small threshold to prevent numerical instability or division by zero.
        """
        super().__init__()
        self.eps = eps

    @abstractmethod
    def fit(self, x: torch.Tensor) -> "BaseScaler":
        """
        Computes scaling statistics (e.g., mean, std, quantiles) from the input tensor.

        Args:
            x (torch.Tensor): Reference feature tensor of shape (Batch, Channels, ...) or (Batch, Features).

        Returns:
            BaseScaler: The fitted scaler instance for method chaining.
        """
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies scaling transformation to the input tensor based on computed statistics.

        Args:
            x (torch.Tensor): Feature tensor matching channel dimensions of fitted data.

        Returns:
            torch.Tensor: Normalized feature tensor of identical shape as input.
        """
        pass

    def _reshape_buffer_for_broadcasting(self, buffer_tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
        """
        Helper method to reshape 1D channel statistics into broadcastable shapes 
        matching arbitrary tensor dimensions: [1, Channels, 1, 1, ...].

        Args:
            buffer_tensor (torch.Tensor): 1D statistics buffer of shape (Channels,).
            target_dim (int): Total rank/dimensions of the input tensor x.

        Returns:
            torch.Tensor: Reshaped buffer tensor ready for broadcasting.
        """
        broadcasting_shape = [1] * target_dim
        broadcasting_shape[1] = -1
        return buffer_tensor.view(broadcasting_shape)


class RobustScaler(BaseScaler):
    """
    PyTorch nn.Module implementation of RobustScaler using quantile statistics.

    Scales features using statistics that are robust to outliers by subtracting 
    the median and dividing by the Interquantile Range (IQR).

    Designed to be integrated directly into PyTorch model wrappers. Statistics 
    are stored as persistent buffers, allowing them to be saved/loaded via `state_dict`.
    """

    def __init__(self, quantile_range: Tuple[float, float] = (1.0, 99.0), eps: float = 1e-7):
        """
        Args:
            quantile_range (Tuple[float, float]): Lower and upper quantile bounds in percentage (0 to 100).
            eps (float): Small threshold value to prevent division by zero or near-zero scaling factors.
        """
        super().__init__()
        self.low_perc = quantile_range[0] / 100.0
        self.high_perc = quantile_range[1] / 100.0
        self.eps = eps

        # Register non-trainable state statistics as buffers
        # These will automatically move across devices (CPU/GPU) and save with state_dict
        self.register_buffer("median", None, persistent=True)
        self.register_buffer("scale", None, persistent=True)

    @torch.no_grad()
    def fit(self, x: torch.Tensor) -> "RobustScaler":
        """
        Computes the median and interquantile range (IQR) across feature channels.

        Args:
            x (torch.Tensor): Feature tensor of shape (Batch, Channels, ...) or (Batch, Features).

        Returns:
            RobustScaler: The fitted scaler instance.
        """
        # Flatten spatial/temporal dimensions while isolating the channel dimension (dim=1)
        if x.dim() > 2:
            # Move channel dimension to end and flatten remaining axes -> (N, Channels)
            x_flat = x.transpose(1, -1).reshape(-1, x.shape[1])
        else:
            x_flat = x

        # Quantile calculations across samples
        q_targets = torch.tensor(
            [self.low_perc, 0.5, self.high_perc], 
            device=x.device, 
            dtype=x.dtype
        )
        
        q = torch.quantile(x_flat, q_targets, dim=0)

        # Extract median and range (IQR)
        computed_median = q[1]
        computed_scale = q[2] - q[0]

        # Prevent zero scale values using epsilon thresholding
        computed_scale = torch.where(
            computed_scale < self.eps, 
            torch.ones_like(computed_scale), 
            computed_scale
        )

        # Store persistent states as registered buffers
        self.register_buffer("median", computed_median, persistent=True)
        self.register_buffer("scale", computed_scale, persistent=True)

        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies scaling to the input tensor based on fitted median and scale.

        Args:
            x (torch.Tensor): Input feature tensor matching channel dimensions of fitted data.

        Returns:
            torch.Tensor: Normalized feature tensor of identical shape as input.
        """
        if self.median is None or self.scale is None:
            raise RuntimeError("❌ Scale Error: RobustScaler must be fitted using `.fit(x)` before calling forward pass!")

        # Reshape buffers to broadcast correctly across input tensor dimensions
        # Shape becomes [1, Channels, 1, 1, ...] matching x.dim()
        broadcasting_shape = [1] * x.dim()
        broadcasting_shape[1] = -1

        median = self.median.view(broadcasting_shape)
        scale = self.scale.view(broadcasting_shape)

        return (x - median) / scale


class StandardScaler(BaseScaler):
    """
    PyTorch nn.Module implementation of StandardScaler.

    Standardizes features by subtracting the mean and scaling to unit variance (std=1).
    Designed to be integrated directly into PyTorch model wrappers. Statistics are stored 
    as persistent buffers, allowing them to be saved/loaded seamlessly via `state_dict`.
    """

    def __init__(self, eps: float = 1e-7):
        """
        Args:
            eps (float): Small threshold value to prevent division by zero for constant features.
        """
        super().__init__()
        self.eps = eps

        # Register non-trainable state statistics as buffers
        self.register_buffer("mean", None, persistent=True)
        self.register_buffer("std", None, persistent=True)

    @torch.no_grad()
    def fit(self, x: torch.Tensor) -> "StandardScaler":
        """
        Computes the mean and standard deviation across feature channels.

        Args:
            x (torch.Tensor): Feature tensor of shape (Batch, Channels, ...) or (Batch, Features).

        Returns:
            StandardScaler: The fitted scaler instance.
        """
        # Flatten spatial/temporal dimensions while isolating the channel dimension (dim=1)
        if x.dim() > 2:
            x_flat = x.transpose(1, -1).reshape(-1, x.shape[1])
        else:
            x_flat = x

        # Calculate mean and standard deviation across sample dimension
        computed_mean = torch.mean(x_flat, dim=0)
        computed_std = torch.std(x_flat, dim=0, unbiased=True)

        # Prevent zero division using epsilon thresholding
        computed_std = torch.where(
            computed_std < self.eps, 
            torch.ones_like(computed_std), 
            computed_std
        )

        # Store persistent states as registered buffers
        self.register_buffer("mean", computed_mean, persistent=True)
        self.register_buffer("std", computed_std, persistent=True)

        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies standard scaling to the input tensor based on fitted mean and std.

        Args:
            x (torch.Tensor): Input feature tensor matching channel dimensions of fitted data.

        Returns:
            torch.Tensor: Standardized feature tensor of identical shape as input.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("❌ Scale Error: StandardScaler must be fitted using `.fit(x)` before calling forward pass!")

        # Reshape buffers to broadcast correctly across input tensor dimensions
        broadcasting_shape = [1] * x.dim()
        broadcasting_shape[1] = -1

        mean = self.mean.view(broadcasting_shape)
        std = self.std.view(broadcasting_shape)

        return (x - mean) / std


class MinMaxScaler(BaseScaler):
    """
    PyTorch nn.Module implementation of MinMaxScaler.

    Transforms features by scaling each feature to a given range [feature_range[0], feature_range[1]].
    Designed to be integrated directly into PyTorch model wrappers. Statistics are stored 
    as persistent buffers, allowing them to be saved/loaded seamlessly via `state_dict`.
    """

    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0), eps: float = 1e-7):
        """
        Args:
            feature_range (Tuple[float, float]): Desired range of transformed data (min_val, max_val).
            eps (float): Small threshold value to prevent division by zero when min equals max.
        """
        super().__init__()
        self.min_val = feature_range[0]
        self.max_val = feature_range[1]
        self.eps = eps

        # Register non-trainable state statistics as buffers
        self.register_buffer("data_min", None, persistent=True)
        self.register_buffer("data_max", None, persistent=True)

    @torch.no_grad()
    def fit(self, x: torch.Tensor) -> "MinMaxScaler":
        """
        Computes the minimum and maximum values across feature channels.

        Args:
            x (torch.Tensor): Feature tensor of shape (Batch, Channels, ...) or (Batch, Features).

        Returns:
            MinMaxScaler: The fitted scaler instance.
        """
        # Flatten spatial/temporal dimensions while isolating the channel dimension (dim=1)
        if x.dim() > 2:
            x_flat = x.transpose(1, -1).reshape(-1, x.shape[1])
        else:
            x_flat = x

        # Calculate minimum and maximum across sample dimension
        computed_min = torch.amin(x_flat, dim=0)
        computed_max = torch.amax(x_flat, dim=0)

        # Store persistent states as registered buffers
        self.register_buffer("data_min", computed_min, persistent=True)
        self.register_buffer("data_max", computed_max, persistent=True)

        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Min-Max scaling to the input tensor based on fitted minimum and maximum.

        Args:
            x (torch.Tensor): Input feature tensor matching channel dimensions of fitted data.

        Returns:
            torch.Tensor: Scaled feature tensor mapped to feature_range of identical shape as input.
        """
        if self.data_min is None or self.data_max is None:
            raise RuntimeError("❌ Scale Error: MinMaxScaler must be fitted using `.fit(x)` before calling forward pass!")

        # Reshape buffers to broadcast correctly across input tensor dimensions
        broadcasting_shape = [1] * x.dim()
        broadcasting_shape[1] = -1

        data_min = self.data_min.view(broadcasting_shape)
        data_max = self.data_max.view(broadcasting_shape)

        # Calculate range scale
        data_range = data_max - data_min
        data_range = torch.where(
            data_range < self.eps, 
            torch.ones_like(data_range), 
            data_range
        )

        # Scale to [0, 1] range first
        x_std = (x - data_min) / data_range

        # Scale to target feature_range [min_val, max_val]
        x_scaled = x_std * (self.max_val - self.min_val) + self.min_val

        return x_scaled
"""PyTorch ops."""
import torch
import torch.nn as nn

# Helper function to initialize weights with normal distribution
def init_normal_weights(module, std=0.05):
    if hasattr(module, 'weight'):
        nn.init.normal_(module.weight, std=std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.zeros_(module.bias)

# Create functions similar to the TensorFlow lambdas
def dense(inputs, units):
    layer = nn.Linear(inputs.size(-1), units)
    init_normal_weights(layer)
    return layer(inputs)

def conv2d(inputs, filters, kernel_size, stride):
    layer = nn.Conv2d(inputs.size(1), filters, kernel_size, stride)
    init_normal_weights(layer)
    return layer(inputs)

def conv3d(inputs, filters, kernel_size, stride):
    layer = nn.Conv3d(inputs.size(1), filters, kernel_size, stride)
    init_normal_weights(layer)
    return layer(inputs)

def tconv2d(inputs, filters, kernel_size, stride):
    layer = nn.ConvTranspose2d(inputs.size(1), filters, kernel_size, stride)
    init_normal_weights(layer)
    return layer(inputs)

def tconv3d(inputs, filters, kernel_size, stride):
    layer = nn.ConvTranspose3d(inputs.size(1), filters, kernel_size, stride)
    init_normal_weights(layer)
    return layer(inputs)

def get_normalization(norm_type, training=None):
    """Return the normalization function."""
    if norm_type == 'batch_norm':
        def apply_batch_norm(x):
            # Create appropriate batch norm based on input dimensions
            training_mode = training if training is not None else True
            if x.dim() == 5:  # 3D input (B, C, D, H, W)
                bn = nn.BatchNorm3d(x.size(1))
            elif x.dim() == 4:  # 2D input (B, C, H, W)
                bn = nn.BatchNorm2d(x.size(1))
            else:  # 1D input
                bn = nn.BatchNorm1d(x.size(1))
            bn.training = training_mode
            return bn(x)
        return apply_batch_norm
    
    elif norm_type == 'layer_norm':
        def apply_layer_norm(x):
            # Normalize over all dims except batch
            normalized_shape = tuple(x.size()[1:])
            ln = nn.LayerNorm(normalized_shape)
            return ln(x)
        return apply_layer_norm
    
    elif norm_type is None or norm_type == '':
        return lambda x: x
    
    else:
        raise ValueError("Unrecognizable normalization type.")
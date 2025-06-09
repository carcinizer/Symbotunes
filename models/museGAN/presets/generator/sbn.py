"""This file defines the network architecture for the generator using PyTorch."""
import torch
import torch.nn as nn
from ..ops import tconv3d, get_normalization
from ..binary_ops import binary_stochastic_ST

NORMALIZATION = 'batch_norm'  # 'batch_norm', 'layer_norm'
ACTIVATION = nn.ReLU  # nn.ReLU, nn.LeakyReLU, torch.tanh, torch.sigmoid

class Generator(nn.Module):
    def __init__(self, n_tracks):
        super().__init__()
        self.n_tracks = n_tracks
        self.norm_type = NORMALIZATION
        self.activation = ACTIVATION()

    def forward(self, x, condition=None, training=True, slope=1.0):
        norm = get_normalization(self.norm_type, training)
        def tconv_layer(i, f, k, s):
            return self.activation(norm(tconv3d(i, f, k, s)))

        # expand latent vector to (N, C, D, H, W)
        h = x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # Shared network
        h = tconv_layer(h, 512, (4, 1, 1), (4, 1, 1))
        h = tconv_layer(h, 256, (1, 4, 3), (1, 4, 3))
        h = tconv_layer(h, 128, (1, 4, 3), (1, 4, 2))

        # Pitch-time private network
        s1 = [tconv_layer(h, 32, (1, 1, 12), (1, 1, 12)) for _ in range(self.n_tracks)]
        s1 = [tconv_layer(s1[i], 16, (1, 3, 1), (1, 3, 1)) for i in range(self.n_tracks)]

        # Time-pitch private network
        s2 = [tconv_layer(h, 32, (1, 3, 1), (1, 3, 1)) for _ in range(self.n_tracks)]
        s2 = [tconv_layer(s2[i], 16, (1, 1, 12), (1, 1, 12)) for i in range(self.n_tracks)]

        # concatenate along channel dimension
        h_list = [torch.cat((s1[i], s2[i]), dim=1) for i in range(self.n_tracks)]

        # Merged private network
        h_merge = [norm(tconv3d(h_list[i], 1, (1, 1, 1), (1, 1, 1))) for i in range(self.n_tracks)]
        h_merge = torch.cat(h_merge, dim=1)

        # Binary activation
        h_binary, preactivated = binary_stochastic_ST(h_merge, slope, False, True)

        # Scale and shift from [0, 1] to [-1, 1]
        h_binary = h_binary * 2. - 1.
        preactivated = preactivated * 2. - 1.

        return h_binary, preactivated

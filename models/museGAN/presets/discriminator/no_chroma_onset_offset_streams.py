"""This file defines the network architecture for the discriminator."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..ops import dense, conv3d, get_normalization

NORMALIZATION = None # 'batch_norm', 'layer_norm', None
ACTIVATION = F.leaky_relu # relu, leaky_relu, tanh, sigmoid

class Discriminator(nn.Module):
    def __init__(self, n_tracks, beat_resolution=None, name='Discriminator'):
        super().__init__()
        self.n_tracks = n_tracks
        self.beat_resolution = beat_resolution
        self.name = name

    def forward(self, tensor_in, condition=None, training=None):
        norm = get_normalization(NORMALIZATION, training)
        conv_layer = lambda x, f, k, s: F.leaky_relu(norm(conv3d(x, f, k, s)), negative_slope=0.2)

        h = tensor_in

        # Pitch-time private network
        with torch.no_grad():
            s1 = [conv_layer(h, 16, (1, 1, 12), (1, 1, 12))      # 4, 48, 7
                  for _ in range(self.n_tracks)]
            s1 = [conv_layer(s1[i], 32, (1, 3, 1), (1, 3, 1))    # 4, 16, 7
                  for i in range(self.n_tracks)]

        # Time-pitch private network
        with torch.no_grad():
            s2 = [conv_layer(h, 16, (1, 3, 1), (1, 3, 1))        # 4, 16, 84
                  for _ in range(self.n_tracks)]
            s2 = [conv_layer(s2[i], 32, (1, 1, 12), (1, 1, 12))  # 4, 16, 7
                  for i in range(self.n_tracks)]

        h = [torch.cat((s1[i], s2[i]), dim=-1) for i in range(self.n_tracks)]

        # Merged private network
        with torch.no_grad():
            h = [conv_layer(h[i], 64, (1, 1, 1), (1, 1, 1))      # 4, 16, 7
                 for i in range(self.n_tracks)]

        h = torch.cat(h, dim=-1)

        # Shared network
        with torch.no_grad():
            h = conv_layer(h, 128, (1, 4, 3), (1, 4, 2))         # 4, 4, 3
            h = conv_layer(h, 256, (1, 4, 3), (1, 4, 3))         # 4, 1, 1

        # Merge all streams
        with torch.no_grad():
            h = conv_layer(h, 512, (2, 1, 1), (1, 1, 1))         # 3, 1, 1

        h = h.view(-1, h.size(-1))
        h = dense(h, 1)

        return h

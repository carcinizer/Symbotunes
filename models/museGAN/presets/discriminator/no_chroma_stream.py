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

        # compute onset/offset feature
        padded = F.pad(tensor_in[:, :, :-1], pad=(0,0, 0,0, 1,0))
        on_off_set = (tensor_in - padded).sum(dim=3, keepdim=True)

        # pitch_time_private
        s1 = [conv_layer(h, 16, (1, 1, 12), (1, 1, 12))      # 4, 48, 7
              for _ in range(self.n_tracks)]
        s1 = [conv_layer(s1[i], 32, (1, 3, 1), (1, 3, 1))    # 4, 16, 7
              for i in range(self.n_tracks)]

        # time_pitch_private
        s2 = [conv_layer(h, 16, (1, 3, 1), (1, 3, 1))        # 4, 16, 84
              for _ in range(self.n_tracks)]
        s2 = [conv_layer(s2[i], 32, (1, 1, 12), (1, 1, 12))  # 4, 16, 7
              for i in range(self.n_tracks)]

        h = [torch.cat((s1[i], s2[i]), dim=-1) for i in range(self.n_tracks)]

        # merged_private
        h = [conv_layer(h[i], 64, (1, 1, 1), (1, 1, 1))      # 4, 16, 7
             for i in range(self.n_tracks)]

        h = torch.cat(h, dim=-1)

        # shared
        h = conv_layer(h, 128, (1, 4, 3), (1, 4, 2))         # 4, 4, 3
        h = conv_layer(h, 256, (1, 4, 3), (1, 4, 3))         # 4, 1, 1

        # on_off_set stream
        o = conv_layer(on_off_set, 16, (1, 3, 1), (1, 3, 1)) # 4, 16, 1
        o = conv_layer(o, 32, (1, 4, 1), (1, 4, 1))          # 4, 4, 1
        o = conv_layer(o, 64, (1, 4, 1), (1, 4, 1))          # 4, 1, 1

        h = torch.cat((h, o), dim=-1)

        # merged
        h = conv_layer(h, 512, (2, 1, 1), (1, 1, 1))         # 3, 1, 1

        h = h.view(-1, h.size(-1))
        h = dense(h, 1)

        return h

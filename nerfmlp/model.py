import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.freq_bands = 2.0 ** torch.arange(num_freqs)

    def forward(self, x):
        # x: (..., input_dim)
        out = [x] if self.include_input else []
        for freq in self.freq_bands:
            out.append(torch.sin(freq * torch.pi * x))
            out.append(torch.cos(freq * torch.pi * x))
        return torch.cat(out, dim=-1)

class NeRFMLP(nn.Module):
    def __init__(self, D=8, W=256, input_ch=63, input_ch_views=27, skips=[4],
                 use_viewdirs=True, output_ch=4):
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(1, D)]
        )
        self.sigma_linear = nn.Linear(W, 1)
        self.feature_linear = nn.Linear(W, W)

        if use_viewdirs:
            self.view_linear = nn.Linear(W + input_ch_views, 128)
            self.rgb_linear = nn.Linear(128, 3)
        else:
            self.rgb_linear = nn.Linear(W, 3)

    def forward(self, x, viewdirs=None):
        # x: (N_rays*N_samples, input_ch)
        # viewdirs: (N_rays, input_ch_views)
        h = x
        for i, l in enumerate(self.pts_linears):
            if i in self.skips:
                h = torch.cat([x, h], -1)
            h = l(h)
            h = F.relu(h)
        sigma = self.sigma_linear(h)
        feature = self.feature_linear(h)
        if self.use_viewdirs and viewdirs is not None:
            h = torch.cat([feature, viewdirs], -1)
            h = self.view_linear(h)
            h = F.relu(h)
            rgb = self.rgb_linear(h)
        else:
            rgb = self.rgb_linear(feature)
        outputs = torch.cat([rgb, sigma], -1)  # (N_rays*N_samples, 4)
        return outputs

    def load_from_numpy(self, np_arrays):
        # Load weights from a list of numpy arrays in the order found in the .npy files
        idx = 0
        # Input and hidden layers
        for l in self.pts_linears:
            l.weight.data.copy_(torch.from_numpy(np_arrays[idx].T))
            l.bias.data.copy_(torch.from_numpy(np_arrays[idx+1]))
            idx += 2
        # Feature and sigma
        self.feature_linear.weight.data.copy_(torch.from_numpy(np_arrays[idx].T))
        self.feature_linear.bias.data.copy_(torch.from_numpy(np_arrays[idx+1]))
        idx += 2
        self.sigma_linear.weight.data.copy_(torch.from_numpy(np_arrays[idx].T))
        self.sigma_linear.bias.data.copy_(torch.from_numpy(np_arrays[idx+1]))
        idx += 2
        # Viewdir and rgb
        if self.use_viewdirs:
            self.view_linear.weight.data.copy_(torch.from_numpy(np_arrays[idx].T))
            self.view_linear.bias.data.copy_(torch.from_numpy(np_arrays[idx+1]))
            idx += 2
        self.rgb_linear.weight.data.copy_(torch.from_numpy(np_arrays[idx].T))
        self.rgb_linear.bias.data.copy_(torch.from_numpy(np_arrays[idx+1])) 
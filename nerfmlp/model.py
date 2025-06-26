import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs, include_input=True, log_sampling=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.log_sampling = log_sampling
        
        if log_sampling:
            # Match official implementation exactly: tf.linspace(0., max_freq, N_freqs)
            # This creates frequencies: [2^0, 2^1, 2^2, ..., 2^(num_freqs-1)]
            self.freq_bands = 2.0 ** torch.linspace(0., num_freqs-1, num_freqs)
        else:
            # Linear sampling: tf.linspace(2.**0., 2.**max_freq, N_freqs)
            self.freq_bands = torch.linspace(2.**0, 2.**(num_freqs-1), num_freqs)

    def forward(self, x):
        # x: (..., input_dim)
        out = [x] if self.include_input else []
        for freq in self.freq_bands:
            out.append(torch.sin(freq * x))  # Match official - no pi multiplication
            out.append(torch.cos(freq * x))
        return torch.cat(out, dim=-1)

class NeRFMLP(nn.Module):
    def __init__(self, D=8, W=256, input_ch=63, input_ch_views=27, skips=[5],
                 use_viewdirs=True, output_ch=4):
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        # Main MLP layers - REVERT: skip at layer 5 (index 5) to match saved weights
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] +                    # Layer 0
            [nn.Linear(W, W) for _ in range(1, 5)] +      # Layers 1-4  
            [nn.Linear(W + input_ch, W)] +                # Layer 5 (skip layer)
            [nn.Linear(W, W) for _ in range(6, D)]        # Layers 6-7
        )
        
        if use_viewdirs:
            # Match official architecture: separate sigma and bottleneck branches
            self.sigma_linear = nn.Linear(W, 1)  # Alpha/sigma branch
            self.bottleneck_linear = nn.Linear(W, 256)  # Bottleneck for viewdirs
            # Viewdirs branch: bottleneck + viewdirs -> 1 hidden layer (128) -> RGB
            self.view_linear = nn.Linear(256 + input_ch_views, W//2)  # W//2 = 128
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x, viewdirs=None):
        # x: (N_rays*N_samples, input_ch)
        # viewdirs: (N_rays*N_samples, input_ch_views)
        h = x
        for i, l in enumerate(self.pts_linears):
            if i == 5:  # REVERT: Skip at layer 5 to match saved weights
                h = torch.cat([x, h], -1)
            h = l(h)
            h = F.relu(h)
        
        if self.use_viewdirs and viewdirs is not None:
            # Match official architecture exactly
            sigma = self.sigma_linear(h)  # Alpha/sigma branch
            bottleneck = self.bottleneck_linear(h)  # Bottleneck branch
            # Concatenate bottleneck with viewdirs
            h = torch.cat([bottleneck, viewdirs], -1)
            h = self.view_linear(h)
            h = F.relu(h)
            rgb = self.rgb_linear(h)
            # Concatenate RGB and sigma (sigma last, matching official)
            outputs = torch.cat([rgb, sigma], -1)
        else:
            outputs = self.output_linear(h)
        
        return outputs

    def load_from_numpy(self, np_arrays):
        # Load weights from a list of numpy arrays in the order found in the .npy files
        # Official order (FIXED to match skip at layer 4):
        # 0-15: main MLP (8 layers) - with skip at layer 4
        # 16-17: bottleneck_linear (256,256), (256,)
        # 18-19: view_linear (283,128), (128,)
        # 20-21: rgb_linear (128,3), (3,)
        # 22-23: sigma_linear (256,1), (1,)
        idx = 0
        # Main MLP layers
        for i, l in enumerate(self.pts_linears):
            print(f"Loading pts_linears[{i}].weight with shape {l.weight.data.shape} from np_arrays[{idx}].T {np_arrays[idx].shape}")
            l.weight.data.copy_(torch.from_numpy(np_arrays[idx].T))
            print(f"Loading pts_linears[{i}].bias with shape {l.bias.data.shape} from np_arrays[{idx+1}].shape {np_arrays[idx+1].shape}")
            l.bias.data.copy_(torch.from_numpy(np_arrays[idx+1]))
            idx += 2
        if self.use_viewdirs:
            # Bottleneck
            print(f"Loading bottleneck_linear.weight with shape {self.bottleneck_linear.weight.data.shape} from np_arrays[{idx}].T {np_arrays[idx].shape}")
            self.bottleneck_linear.weight.data.copy_(torch.from_numpy(np_arrays[idx].T))
            print(f"Loading bottleneck_linear.bias with shape {self.bottleneck_linear.bias.data.shape} from np_arrays[{idx+1}].shape {np_arrays[idx+1].shape}")
            self.bottleneck_linear.bias.data.copy_(torch.from_numpy(np_arrays[idx+1]))
            idx += 2
            # Viewdirs
            print(f"Loading view_linear.weight with shape {self.view_linear.weight.data.shape} from np_arrays[{idx}].T {np_arrays[idx].shape}")
            self.view_linear.weight.data.copy_(torch.from_numpy(np_arrays[idx].T))
            print(f"Loading view_linear.bias with shape {self.view_linear.bias.data.shape} from np_arrays[{idx+1}].shape {np_arrays[idx+1].shape}")
            self.view_linear.bias.data.copy_(torch.from_numpy(np_arrays[idx+1]))
            idx += 2
            # RGB
            print(f"Loading rgb_linear.weight with shape {self.rgb_linear.weight.data.shape} from np_arrays[{idx}].T {np_arrays[idx].shape}")
            self.rgb_linear.weight.data.copy_(torch.from_numpy(np_arrays[idx].T))
            print(f"Loading rgb_linear.bias with shape {self.rgb_linear.bias.data.shape} from np_arrays[{idx+1}].shape {np_arrays[idx+1].shape}")
            self.rgb_linear.bias.data.copy_(torch.from_numpy(np_arrays[idx+1]))
            idx += 2
            # Sigma
            print(f"Loading sigma_linear.weight with shape {self.sigma_linear.weight.data.shape} from np_arrays[{idx}].T {np_arrays[idx].shape}")
            self.sigma_linear.weight.data.copy_(torch.from_numpy(np_arrays[idx].T))
            print(f"Loading sigma_linear.bias with shape {self.sigma_linear.bias.data.shape} from np_arrays[{idx+1}].shape {np_arrays[idx+1].shape}")
            self.sigma_linear.bias.data.copy_(torch.from_numpy(np_arrays[idx+1]))
        else:
            print(f"Loading output_linear.weight with shape {self.output_linear.weight.data.shape} from np_arrays[{idx}].T {np_arrays[idx].shape}")
            self.output_linear.weight.data.copy_(torch.from_numpy(np_arrays[idx].T))
            print(f"Loading output_linear.bias with shape {self.output_linear.bias.data.shape} from np_arrays[{idx+1}].shape {np_arrays[idx+1].shape}")
            self.output_linear.bias.data.copy_(torch.from_numpy(np_arrays[idx+1])) 
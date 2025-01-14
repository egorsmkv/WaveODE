import math
import torch
# import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy import integrate
# from torchdiffeq import odeint

# from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
# from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

def to_flattened_numpy(x):
  """Flatten a torch tensor `x` and convert it to numpy."""
  return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
  """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
  return torch.from_numpy(x.reshape(shape))

def swish(x):
    return x * torch.sigmoid(x)

def calc_diffusion_step_embedding(diffusion_steps, diffusion_step_embed_dim_in):
    """
    Embed a diffusion step $t$ into a higher dimensional space
        E.g. the embedding vector in the 128-dimensional space is
        [sin(t * 10^(0*4/63)), ... , sin(t * 10^(63*4/63)),
         cos(t * 10^(0*4/63)), ... , cos(t * 10^(63*4/63))]
    Parameters:
        diffusion_steps (torch.long tensor, shape=(batchsize, 1)):
                                    diffusion steps for batch data
        diffusion_step_embed_dim_in (int, default=128):
                                    dimensionality of the embedding space for discrete diffusion steps
    Returns:
        the embedding vectors (torch.tensor, shape=(batchsize, diffusion_step_embed_dim_in)):
    """

    assert diffusion_step_embed_dim_in % 2 == 0

    half_dim = diffusion_step_embed_dim_in // 2
    _embed = np.log(10000) / (half_dim - 1)
    _embed = torch.exp(torch.arange(half_dim) * -_embed).cuda()
    _embed = diffusion_steps * _embed
    diffusion_step_embed = torch.cat((torch.sin(_embed),
                                      torch.cos(_embed)), 1)
    return diffusion_step_embed

"""
Below scripts were borrowed from
https://github.com/philsyn/DiffWave-Vocoder/blob/master/WaveNet.py
"""

# dilated conv layer with kaiming_normal initialization
# from https://github.com/ksw0306/FloWaveNet/blob/master/modules.py
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding)
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        out = self.conv(x)
        return out


# conv1x1 layer with zero initialization
# from https://github.com/ksw0306/FloWaveNet/blob/master/modules.py but the scale parameter is removed
class ZeroConv1d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    def forward(self, x):
        out = self.conv(x)
        return out

# every residual block (named residual layer in paper)
# contains one noncausal dilated conv
class ResidualBlock(nn.Module):
    def __init__(self, res_channels, skip_channels, dilation,
                 diffusion_step_embed_dim_out):
        super().__init__()
        self.res_channels = res_channels

        # Use a FC layer for diffusion step embedding
        self.fc_t = nn.Linear(diffusion_step_embed_dim_out, self.res_channels)

        # Dilated conv layer
        self.dilated_conv_layer = Conv(self.res_channels, 2 * self.res_channels,
                                       kernel_size=3, dilation=dilation)

        # Add mel spectrogram upsampler and conditioner conv1x1 layer
        self.upsample_conv2d = nn.ModuleList()
        for s in [16, 16]:
            conv_trans2d = nn.ConvTranspose2d(1, 1, (3, 2 * s),
                                              padding=(1, s // 2),
                                              stride=(1, s))
            conv_trans2d = nn.utils.weight_norm(conv_trans2d)
            nn.init.kaiming_normal_(conv_trans2d.weight)
            self.upsample_conv2d.append(conv_trans2d)

        # 80 is mel bands
        self.mel_conv = Conv(80, 2 * self.res_channels, kernel_size=1)

        # Residual conv1x1 layer, connect to next residual layer
        self.res_conv = nn.Conv1d(res_channels, res_channels, kernel_size=1)
        self.res_conv = nn.utils.weight_norm(self.res_conv)
        nn.init.kaiming_normal_(self.res_conv.weight)

        # Skip conv1x1 layer, add to all skip outputs through skip connections
        self.skip_conv = nn.Conv1d(res_channels, skip_channels, kernel_size=1)
        self.skip_conv = nn.utils.weight_norm(self.skip_conv)
        nn.init.kaiming_normal_(self.skip_conv.weight)

    def forward(self, input_data):
        x, mel_spec, diffusion_step_embed = input_data
        h = x
        batch_size, n_channels, seq_len = x.shape
        assert n_channels == self.res_channels

        # Add in diffusion step embedding
        part_t = self.fc_t(diffusion_step_embed)
        part_t = part_t.view([batch_size, self.res_channels, 1])
        h += part_t

        # Dilated conv layer
        h = self.dilated_conv_layer(h)

        # Upsample spectrogram to size of audio
        mel_spec = torch.unsqueeze(mel_spec, dim=1)
        mel_spec = F.leaky_relu(self.upsample_conv2d[0](mel_spec), 0.4, inplace=False)
        mel_spec = F.leaky_relu(self.upsample_conv2d[1](mel_spec), 0.4, inplace=False)
        mel_spec = torch.squeeze(mel_spec, dim=1)

        assert mel_spec.size(2) >= seq_len
        if mel_spec.size(2) > seq_len:
            mel_spec = mel_spec[:, :, :seq_len]

        mel_spec = self.mel_conv(mel_spec)
        h += mel_spec

        # Gated-tanh nonlinearity
        out = torch.tanh(h[:, :self.res_channels, :]) * torch.sigmoid(h[:, self.res_channels:, :])

        # Residual and skip outputs
        res = self.res_conv(out)
        assert x.shape == res.shape
        skip = self.skip_conv(out)

        # Normalize for training stability
        return (x + res) * math.sqrt(0.5), skip

class ResidualGroup(nn.Module):
    def __init__(self, res_channels, skip_channels, num_res_layers, dilation_cycle,
                 diffusion_step_embed_dim_in,
                 diffusion_step_embed_dim_mid,
                 diffusion_step_embed_dim_out):
        super().__init__()
        self.num_res_layers = num_res_layers
        self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in

        # Use the shared two FC layers for diffusion step embedding
        self.fc_t1 = nn.Linear(diffusion_step_embed_dim_in, diffusion_step_embed_dim_mid)
        self.fc_t2 = nn.Linear(diffusion_step_embed_dim_mid, diffusion_step_embed_dim_out)

        # Stack all residual blocks with dilations 1, 2, ... , 512, ... , 1, 2, ..., 512
        self.residual_blocks = nn.ModuleList()
        for n in range(self.num_res_layers):
            self.residual_blocks.append(
                ResidualBlock(res_channels, skip_channels,
                               dilation=2 ** (n % dilation_cycle),
                               diffusion_step_embed_dim_out=diffusion_step_embed_dim_out))

    def forward(self, input_data):
        x, mel_spectrogram, diffusion_steps = input_data

        # Embed diffusion step t
        diffusion_step_embed = calc_diffusion_step_embedding(
            diffusion_steps, self.diffusion_step_embed_dim_in)
        diffusion_step_embed = swish(self.fc_t1(diffusion_step_embed))
        diffusion_step_embed = swish(self.fc_t2(diffusion_step_embed))

        # Pass all residual layers
        h = x
        skip = 0
        for n in range(self.num_res_layers):
            # Use the output from last residual layer
            h, skip_n = self.residual_blocks[n]((h, mel_spectrogram, diffusion_step_embed))
            # Accumulate all skip outputs
            skip = skip + skip_n

        return skip * math.sqrt(1.0 / self.num_res_layers)


class DiffWave(nn.Module):
    def __init__(self, in_channels, res_channels, skip_channels, out_channels,
                 num_res_layers, dilation_cycle,
                 diffusion_step_embed_dim_in,
                 diffusion_step_embed_dim_mid,
                 diffusion_step_embed_dim_out):
        super().__init__()

        # Initial conv1x1 with relu
        self.init_conv = nn.Sequential(Conv(in_channels, res_channels, kernel_size=1), nn.ReLU(inplace=False))
        # All residual layers
        self.residual_layer = ResidualGroup(res_channels,
                                            skip_channels,
                                            num_res_layers,
                                            dilation_cycle,
                                            diffusion_step_embed_dim_in,
                                            diffusion_step_embed_dim_mid,
                                            diffusion_step_embed_dim_out)
        # Final conv1x1 -> relu -> zeroconv1x1
        self.final_conv = nn.Sequential(Conv(skip_channels, skip_channels, kernel_size=1),
                                        nn.ReLU(inplace=False), ZeroConv1d(skip_channels, out_channels))

    def forward(self, input_data):
        x, condition, diffusion_steps = input_data
        x = self.init_conv(x).clone()
        x = self.residual_layer((x, condition, diffusion_steps))
        return self.final_conv(x)

class Generator(nn.Module):
    
    def __init__(self, hparams):
        super(Generator, self).__init__()
        self.hparams = hparams
        self.T = hparams.T
        self.eps = hparams.eps
        self.noise_scale = hparams.noise_scale
        self.reflow_flag = hparams.reflow_flag
        if self.reflow_flag:
            self.reflow_t_schedule = hparams.reflow_t_schedule
        self.t_max_value = hparams.t_max_value
        self.sigma_t = lambda t: (1. - t) * hparams.sigma_var

        self.generator = DiffWave(in_channels=hparams.in_dim, res_channels=hparams.res_dim, skip_channels=hparams.skip_dim, out_channels=hparams.out_dim,
                                  num_res_layers=hparams.n_res_layers, dilation_cycle=hparams.dilation_cycle,
                                  diffusion_step_embed_dim_in=hparams.diffusion_step_embed_in_dim,
                                  diffusion_step_embed_dim_mid=hparams.diffusion_step_embed_mid_dim,
                                  diffusion_step_embed_dim_out=hparams.diffusion_step_embed_out_dim)

    def remove_weight_norm(self):
        self.generator.remove_weight_norm()

    def forward(self, features, target_features, noise):

        if self.reflow_flag:
            if self.reflow_t_schedule=='t0': ### distill for t = 0 (k=1)
                t = torch.zeros((target_features.shape[0], 1), device=target_features.device) * (self.T - self.eps) + self.eps
            elif self.reflow_t_schedule=='t1': ### reverse distill for t=1 (fast embedding)
                t = torch.ones((target_features.shape[0], 1), device=target_features.device) * (self.T - self.eps) + self.eps
            elif self.reflow_t_schedule=='uniform': ### train new rectified flow with reflow
                t = torch.rand((target_features.shape[0], 1), device=target_features.device) * (self.T - self.eps) + self.eps
            elif type(self.reflow_t_schedule)==int: ### k > 1 distillation
                t = torch.randint(0, self.reflow_t_schedule, (target_features.shape[0], 1), device=target_features.device) * (self.T - eps) / self.reflow_t_schedule + eps
            else:
                assert False, 'Not implemented'
        else:
            ### standard rectified flow loss
            t = torch.rand((target_features.shape[0], 1), device=target_features.device) * (self.T - self.eps) + self.eps

        t_expand = t.view(-1, 1, 1).repeat(1, target_features.shape[1], target_features.shape[2])
        perturbed_data = t_expand * target_features + (1. - t_expand) * noise        
        predicted_score = self.generator((perturbed_data, features, t*self.t_max_value))
        target_score = target_features - noise

        return predicted_score, target_score 

    @torch.jit.ignore
    @torch.no_grad()
    def inference(self, features, sampling_method='euler', sampling_steps=1000):
        init_noise = torch.randn(features.shape[0], 1, features.shape[-1] * self.hparams.hop_size).to(features.device) * self.hparams.noise_scale
        x = init_noise.clone()
        shape = x.shape
        if sampling_method == 'euler':
            print('sampling_method: euler')
            dt = 1./sampling_steps
            for i in range(sampling_steps):
        
                num_t = i /sampling_steps * (self.T - self.eps) + self.eps
                t = torch.ones((shape[0], 1), device=features.device) * num_t
                pred = self.generator((x, features, t*999)) ### Copy from models/utils.py 

                # convert to diffusion models if sampling.sigma_variance > 0.0 while perserving the marginal probability 
                sigma_t = self.sigma_t(num_t)
                pred_sigma = pred + (sigma_t**2)/(2*(self.noise_scale**2)*((1.-num_t)**2)) * (0.5 * num_t * (1.-num_t) * pred - 0.5 * (2.-num_t)*x)

                x = x + pred_sigma * dt + sigma_t * np.sqrt(dt) * torch.randn_like(pred_sigma).to(features.device)

        elif sampling_method == 'rk45':
            print('sampling_method: rk45')
            def ode_func(t, x):
                x = from_flattened_numpy(x, shape).to(features.device).type(torch.float32)
                vec_t = torch.ones((shape[0], 1), device=x.device) * t
                drift = self.generator((x, features, vec_t*999))
                return to_flattened_numpy(drift)
           
            solution = integrate.solve_ivp(ode_func, (self.eps, self.T), to_flattened_numpy(x),
                                     rtol=self.hparams.ode_tol, atol=self.hparams.ode_tol, method='RK45') 


            x = torch.tensor(solution.y[:, -1]).reshape(x.shape).to(x.device).type(torch.float32)

        predicted_audio = x
        return predicted_audio, init_noise

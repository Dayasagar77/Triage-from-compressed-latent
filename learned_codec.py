"""
learned_codec.py — a self-contained learned image codec (pure PyTorch, no C++).

Design (Balle-style factorized-prior codec):
  * GDN-based analysis (g_a) / synthesis (g_s) transforms,
  * scalar quantization (uniform-noise proxy in training, rounding at eval),
  * a learned factorized entropy model (per-channel Gaussian) giving the bit-rate,
  * trained with a rate-distortion loss  L = lambda * 255^2 * MSE + bpp.

Rate is the entropy model's estimated bits (the near-optimal rate an arithmetic
coder achieves); this is the standard rate metric in learned-compression research.
This lets the codec be optimized for BYTES and quality jointly — the reason it can
beat JPEG, which the raw-latent v1 autoencoder could not.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GDN(nn.Module):
    """Generalized Divisive Normalization (Balle 2016), simplified."""
    def __init__(self, ch, inverse=False):
        super().__init__()
        self.inverse = inverse
        self.beta = nn.Parameter(torch.ones(ch))
        self.gamma = nn.Parameter(0.1 * torch.eye(ch))
        self.ch = ch

    def forward(self, x):
        g = self.gamma.abs().view(self.ch, self.ch, 1, 1)
        norm = torch.sqrt(F.conv2d(x * x, g, self.beta.abs()) + 1e-6)
        return x * norm if self.inverse else x / norm


def conv(i, o):   return nn.Conv2d(i, o, 5, stride=2, padding=2)
def deconv(i, o): return nn.ConvTranspose2d(i, o, 5, stride=2, padding=2, output_padding=1)


class FactorizedCodec(nn.Module):
    """224x224 -> 14x14xM latent -> 224x224. Rate from a factorized Gaussian prior."""
    def __init__(self, N=128, M=192):
        super().__init__()
        self.M = M
        self.g_a = nn.Sequential(conv(3, N), GDN(N), conv(N, N), GDN(N),
                                 conv(N, N), GDN(N), conv(N, M))
        self.g_s = nn.Sequential(deconv(M, N), GDN(N, inverse=True), deconv(N, N), GDN(N, inverse=True),
                                 deconv(N, N), GDN(N, inverse=True), deconv(N, 3), nn.Sigmoid())
        self.prior_mean = nn.Parameter(torch.zeros(M))
        self.prior_logscale = nn.Parameter(torch.zeros(M))

    def quantize(self, y):
        if self.training:
            return y + (torch.rand_like(y) - 0.5)   # uniform-noise proxy
        return torch.round(y)

    def rate_bits(self, y_hat):
        mean = self.prior_mean.view(1, -1, 1, 1)
        scale = F.softplus(self.prior_logscale).clamp(min=0.11).view(1, -1, 1, 1)
        d = torch.distributions.Normal(mean, scale)
        p = (d.cdf(y_hat + 0.5) - d.cdf(y_hat - 0.5)).clamp(min=1e-9)
        return (-torch.log2(p)).sum()

    def forward(self, x):
        y = self.g_a(x)
        y_hat = self.quantize(y)
        x_hat = self.g_s(y_hat).clamp(0, 1)
        bits = self.rate_bits(y_hat)
        return {"x_hat": x_hat, "y": y, "y_hat": y_hat, "bits": bits}

    @torch.no_grad()
    def encode_latent(self, x):
        """The transmitted code: quantized latent (for classify-from-latent)."""
        return torch.round(self.g_a(x))

# ******************************************************************************
# Copyright (C) 2024 NVIDIA CORPORATION
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ******************************************************************************
# Adapted for BandTok from NVIDIA Cosmos Tokenizer (https://github.com/NVIDIA/Cosmos-Tokenizer).
# Only the 2D paths used by BandTok are retained. License: Apache-2.0, see LICENSES/LICENSE.apache-2.0.txt
"""The patcher and unpatcher implementation for 2D data.

The idea of Haar wavelet is to compute LL, LH, HL, HH component as two 1D convolutions.
One on the rows and one on the columns.
For example, in 1D signal, we have [a, b], then the low-freq compoenent is [a + b] / 2 and high-freq is [a - b] / 2.
We can use a 1D convolution with kernel [1, 1] and stride 2 to represent the L component.
For H component, we can use a 1D convolution with kernel [1, -1] and stride 2.
Although in principle, we typically only do additional Haar wavelet over the LL component. But here we do it for all
   as we need to support downsampling for more than 2x.
For example, 4x downsampling can be done by 2x Haar and additional 2x Haar, and the shape would be.
   [3, 256, 256] -> [12, 128, 128] -> [48, 64, 64]
"""

import torch
import torch.nn.functional as F
from einops import rearrange

_WAVELETS = {
    "haar": torch.tensor([0.7071067811865476, 0.7071067811865476]),
    "rearrange": torch.tensor([1.0, 1.0]),
}
_PERSISTENT = False


class Patcher(torch.nn.Module):
    """A module to convert image tensors into patches using torch operations.

    The main difference from `class Patching` is that this module implements
    all operations using torch, rather than python or numpy, for efficiency purpose.

    It's bit-wise identical to the Patching module outputs, with the added
    benefit of being torch.jit scriptable.
    """

    def __init__(self, patch_size=1, patch_method="haar"):
        super().__init__()
        self.patch_size = patch_size
        self.patch_method = patch_method
        self.register_buffer("wavelets", _WAVELETS[patch_method], persistent=_PERSISTENT)
        self.range = range(int(torch.log2(torch.tensor(self.patch_size)).item()))
        self.register_buffer(
            "_arange",
            torch.arange(_WAVELETS[patch_method].shape[0]),
            persistent=_PERSISTENT,
        )
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        if self.patch_method == "haar":
            return self._haar(x)
        elif self.patch_method == "rearrange":
            return self._arrange(x)
        else:
            raise ValueError("Unknown patch method: " + self.patch_method)

    def _dwt(self, x, mode="reflect", rescale=False):
        dtype = x.dtype
        h = self.wavelets

        n = h.shape[0]
        g = int(x.shape[1])
        hl = h.flip(0).reshape(1, 1, -1).repeat(g, 1, 1)
        hh = (h * ((-1) ** self._arange)).reshape(1, 1, -1).repeat(g, 1, 1)
        hh = hh.to(dtype=dtype)
        hl = hl.to(dtype=dtype)

        x = F.pad(x, pad=(n - 2, n - 1, n - 2, n - 1), mode=mode).to(dtype)
        xl = F.conv2d(x, hl.unsqueeze(2), groups=g, stride=(1, 2))
        xh = F.conv2d(x, hh.unsqueeze(2), groups=g, stride=(1, 2))
        xll = F.conv2d(xl, hl.unsqueeze(3), groups=g, stride=(2, 1))
        xlh = F.conv2d(xl, hh.unsqueeze(3), groups=g, stride=(2, 1))
        xhl = F.conv2d(xh, hl.unsqueeze(3), groups=g, stride=(2, 1))
        xhh = F.conv2d(xh, hh.unsqueeze(3), groups=g, stride=(2, 1))

        out = torch.cat([xll, xlh, xhl, xhh], dim=1)
        if rescale:
            out = out / 2
        return out

    def _haar(self, x):
        for _ in self.range:
            x = self._dwt(x, rescale=True)
        return x

    def _arrange(self, x):
        x = rearrange(
            x,
            "b c (h p1) (w p2) -> b (c p1 p2) h w",
            p1=self.patch_size,
            p2=self.patch_size,
        ).contiguous()
        return x


class UnPatcher(torch.nn.Module):
    """A module to convert patches into image tensorsusing torch operations.

    The main difference from `class Unpatching` is that this module implements
    all operations using torch, rather than python or numpy, for efficiency purpose.

    It's bit-wise identical to the Unpatching module outputs, with the added
    benefit of being torch.jit scriptable.
    """

    def __init__(self, patch_size=1, patch_method="haar"):
        super().__init__()
        self.patch_size = patch_size
        self.patch_method = patch_method
        self.register_buffer("wavelets", _WAVELETS[patch_method], persistent=_PERSISTENT)
        self.range = range(int(torch.log2(torch.tensor(self.patch_size)).item()))
        self.register_buffer(
            "_arange",
            torch.arange(_WAVELETS[patch_method].shape[0]),
            persistent=_PERSISTENT,
        )
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        if self.patch_method == "haar":
            return self._ihaar(x)
        elif self.patch_method == "rearrange":
            return self._iarrange(x)
        else:
            raise ValueError("Unknown patch method: " + self.patch_method)

    def _idwt(self, x, wavelet="haar", mode="reflect", rescale=False):
        dtype = x.dtype
        h = self.wavelets
        n = h.shape[0]

        g = int(x.shape[1] // 4)
        hl = h.flip([0]).reshape(1, 1, -1).repeat([g, 1, 1])
        hh = (h * ((-1) ** self._arange)).reshape(1, 1, -1).repeat(g, 1, 1)
        hh = hh.to(dtype=dtype)
        hl = hl.to(dtype=dtype)

        xll, xlh, xhl, xhh = torch.chunk(x.to(dtype), 4, dim=1)

        # Inverse transform.
        yl = torch.nn.functional.conv_transpose2d(xll, hl.unsqueeze(3), groups=g, stride=(2, 1), padding=(n - 2, 0))
        yl += torch.nn.functional.conv_transpose2d(xlh, hh.unsqueeze(3), groups=g, stride=(2, 1), padding=(n - 2, 0))
        yh = torch.nn.functional.conv_transpose2d(xhl, hl.unsqueeze(3), groups=g, stride=(2, 1), padding=(n - 2, 0))
        yh += torch.nn.functional.conv_transpose2d(xhh, hh.unsqueeze(3), groups=g, stride=(2, 1), padding=(n - 2, 0))
        y = torch.nn.functional.conv_transpose2d(yl, hl.unsqueeze(2), groups=g, stride=(1, 2), padding=(0, n - 2))
        y += torch.nn.functional.conv_transpose2d(yh, hh.unsqueeze(2), groups=g, stride=(1, 2), padding=(0, n - 2))

        if rescale:
            y = y * 2
        return y

    def _ihaar(self, x):
        for _ in self.range:
            x = self._idwt(x, "haar", rescale=True)
        return x

    def _iarrange(self, x):
        x = rearrange(
            x,
            "b (c p1 p2) h w -> b c (h p1) (w p2)",
            p1=self.patch_size,
            p2=self.patch_size,
        )
        return x

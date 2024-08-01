import torch
from torch import nn
import torch_dct as dct
from pytorch_wavelets import DWT1D, IDWT1D

# Custom imports
from methods import reshape_tensor, perform_STDFT


# In this file, the different transformations--and their inversions--will be implemented as layers.
# To allow LRP to be calculated for complex tensors, please adjust Captum's Propagation Rule class
# In it's _create_backward_hook_output, torch.sign(outputs) must be changed to torch.sgn(outputs)
# This allows the sign function to be applied to complex tensors as well.

class DFT_Layer(nn.Module):
    def __init__(self):
        super(DFT_Layer, self).__init__()

    def forward(self, x):
        X_dft = torch.fft.fft(x, norm="ortho")  # Normalization to maintain energy across the transform and thus ensure
                                                # accurate reconstruction
        return X_dft


class IDFT_Layer(nn.Module):
    def __init__(self):
        super(IDFT_Layer, self).__init__()

    def forward(self, X_dft):
        x = torch.fft.ifft(X_dft, norm="ortho")

        return reshape_tensor(x.real)  # discard imaginary part



class STDFT_Layer(nn.Module):
    def __init__(self):
        super(STDFT_Layer, self).__init__()

    def forward(self, x):
        X_stdft = perform_STDFT(x.squeeze())
        return X_stdft


class ISTDFT_Layer(nn.Module):
    def __init__(self):
        super(ISTDFT_Layer, self).__init__()

    def forward(self, X_stdft):
        x = perform_STDFT(X_stdft, inverse=True)

        return reshape_tensor(x)


class DCT_Layer(nn.Module):
    def __init__(self):
        super(DCT_Layer, self).__init__()

    def forward(self, x):
        x_dct = dct.dct(x, norm='ortho')

        return x_dct


class IDCT_Layer(nn.Module):
    def __init__(self):
        super(IDCT_Layer, self).__init__()

    def forward(self, x_dct):
        x = dct.idct(x_dct, norm='ortho')

        return reshape_tensor(x)


class DWT_Layer(nn.Module):
    def __init__(self):
        super(DWT_Layer, self).__init__()

    def forward(self, x):
        xfm = DWT1D(J=3, mode='zero', wave='db3')

        x = reshape_tensor(x)
        Yl, Yh = xfm(x)
        combined_tensor = torch.cat((Yl, Yh[0], Yh[1], Yh[2]), dim=2)

        return combined_tensor


class IDWT_Layer(nn.Module):
    def __init__(self):
        super(IDWT_Layer, self).__init__()

    def forward(self, combined_tensor):
        ifm = IDWT1D(mode='zero', wave='db3')

        # Reconstruct Yl and Yh from combined_tensor (indexes calculated using dummy input earlier)
        Yl_reconstructed = combined_tensor[:, :, :1004]
        Yh_reconstructed = [combined_tensor[:, :, 1004:5006], combined_tensor[:, :, 5006:7009],
                            combined_tensor[:, :, 7009:8013]]

        x = ifm((Yl_reconstructed, Yh_reconstructed))

        return reshape_tensor(x)
# torch
import torch
from torch import nn


class Dis(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self,):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        input_nc=1
        ndf=64
        n_layers=3
        norm_layer=nn.InstanceNorm3d

        super(Dis, self).__init__()
        use_bias = (norm_layer != nn.BatchNorm2d)

        kw = 4
        padw = 1
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
    
    def gen_loss(self, fake):
        out_fake_list = self.forward(fake)
        loss = 0
        for _, (out_fake) in enumerate(out_fake_list):
            loss += torch.mean((out_fake - 1)**2)
        return loss

    def dis_loss(self, fake, true):
        out_fake_list = self.forward(fake)
        out_true_list = self.forward(true)
        loss = 0
        for _, (out_fake, out_true) in enumerate(zip(out_fake_list, out_true_list)):
            loss += torch.mean((out_fake - 0)**2) + \
                torch.mean((out_true - 1)**2)
        return loss
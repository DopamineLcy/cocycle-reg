# system
import os

# torch
import torch
import torch.nn as nn

# local
from nets import Gen
from nets import Dis
from nets.voxelmorph.networks import VxmDense as Reg
from utils import Transformer_3D
from utils import smooth_loss


class Trainer(nn.Module):
    def __init__(self, cfg):
        super(Trainer, self).__init__()
        self.cfg = cfg
        self.create_models()
        self.setup_optimizers()
        self.sim_loss = torch.nn.L1Loss()
        self.run_diss = 0
        self.gen_gan_loss = (torch.ones((1))*0.5).cuda()

    def create_models(self):
        self.gen_a2b = Gen()
        self.gen_b2a = Gen()
        self.dis_a2b = Dis()
        self.dis_b2a = Dis()
        self.reg = Reg()

    def setup_optimizers(self):
        # params
        dis_param = list(self.dis_a2b.parameters()) + \
            list(self.dis_b2a.parameters())
        gen_param = list(self.gen_a2b.parameters()) + \
            list(self.gen_b2a.parameters())
        reg_param = list(self.reg.parameters())

        self.dis_opt = torch.optim.Adam(
            [p for p in dis_param if p.requires_grad],
            lr=1e-4,
            betas=(0.5, 0.999),
            weight_decay=1e-4,
        )

        self.gen_opt = torch.optim.Adam(
            [p for p in gen_param if p.requires_grad],
            lr=1e-4,
            betas=(0.5, 0.999),
            weight_decay=1e-4,
        )
        self.reg_opt = torch.optim.Adam(
            [p for p in reg_param if p.requires_grad],
            lr=1e-4,
            betas=(0.5, 0.999),
            weight_decay=1e-4,
        )

    def set_input(self, A, B,epoch,iter,txt_path):
        self.A = A
        self.B = B
        self.epoch = epoch
        self.iter = iter
        self.txt_path = txt_path

    def trans(self, img, flow):
        return Transformer_3D().forward(img, flow)

    def forward(self):
        two_deformations = self.reg(self.A, self.B)
        self.smooth_loss_A = smooth_loss(two_deformations[0])
        self.smooth_loss_B = smooth_loss(two_deformations[1])
        self.flow_ab, self.flow_ba = two_deformations[0], two_deformations[1]
        # A pipline
        self.A_g = self.gen_a2b(self.A)
        self.A_gw = self.trans(self.A_g,self.flow_ab)
        self.A_gwr = self.gen_b2a(self.A_gw)
        self.A_gwrb = self.trans(self.A_gwr,self.flow_ba)
        # B pipline
        self.B_g = self.gen_b2a(self.B)
        self.B_gw = self.trans(self.B_g,self.flow_ba)
        self.B_gwr = self.gen_a2b(self.B_gw)
        self.B_gwrb = self.trans(self.B_gwr,self.flow_ab)

    def backward_dis(self):
        dis_loss_AB = self.dis_a2b.dis_loss(self.A_g.detach(), self.B)
        dis_loss_BA = self.dis_b2a.dis_loss(self.B_g.detach(), self.A)
        self.dis_loss = 0.5 * self.cfg['gen_w']*(dis_loss_AB+dis_loss_BA)
        self.dis_loss.backward()

    def backward_gen_reg(self):
        # sim loss
        self.gen_sim_loss = self.cfg['sim_w']*(self.sim_loss(
            self.A_gw, self.B)+self.sim_loss(self.B_gw, self.A))
        # gan loss
        gen_loss_AB = self.dis_a2b.gen_loss(self.A_g)
        gen_loss_BA = self.dis_b2a.gen_loss(self.B_g)
        self.gen_gan_loss = self.cfg['gen_w']*(gen_loss_AB+gen_loss_BA)
        # cycle loss
        gen_loss_AA = self.sim_loss(self.A, self.A_gwrb)
        gen_loss_BB = self.sim_loss(self.B, self.B_gwrb)
        self.gen_cycle_loss = self.cfg['cycle_w'] * \
            (gen_loss_AA+gen_loss_BB)
        self.smooth_loss = self.cfg['smooth_w'] * \
            (self.smooth_loss_A+self.smooth_loss_B)
        self.gen_loss = self.gen_sim_loss + self.gen_gan_loss + self.gen_cycle_loss + self.smooth_loss
        self.gen_loss.backward()

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def optimize(self):
        self.forward()
        # backward dis
        self.set_requires_grad([self.reg, self.gen_a2b, self.gen_b2a], False)
        self.dis_opt.zero_grad()
        self.backward_dis()
        if self.epoch!=0:
            nn.utils.clip_grad_norm_(list(self.dis_a2b.parameters()), max_norm=20, norm_type=2)
            nn.utils.clip_grad_norm_(list(self.dis_b2a.parameters()), max_norm=20, norm_type=2)
        self.dis_opt.step()
        self.set_requires_grad([self.reg, self.gen_a2b, self.gen_b2a], True)
        # backward gen and reg
        self.set_requires_grad([self.dis_a2b, self.dis_b2a], False)
        self.reg_opt.zero_grad()
        self.gen_opt.zero_grad()
        self.backward_gen_reg()
        if self.epoch!=0:
            nn.utils.clip_grad_norm_(list(self.gen_a2b.parameters()), max_norm=20, norm_type=2)
            nn.utils.clip_grad_norm_(list(self.gen_b2a.parameters()), max_norm=20, norm_type=2)
        self.reg_opt.step()
        self.gen_opt.step()
        self.set_requires_grad([self.dis_a2b, self.dis_b2a], True)

    def cal_dice(self, A, B):
        A = A.round()
        B = B.round()
        num = A.size(0)
        A_flat = A.view(num, -1)
        B_flat = B.view(num, -1)
        inter = (A_flat * B_flat).sum(1)
        return (2.0 * inter) / (A_flat.sum(1) + B_flat.sum(1))

    def validate(self, A, B, A_seg, B_seg):
        with torch.no_grad():
            two_deformations = self.reg(A, B)
            flow_ab, flow_ba = two_deformations[0], two_deformations[1]
            A_seg_warped = (Transformer_3D().forward(A_seg, flow_ab))
            B_seg_warped = (Transformer_3D().forward(B_seg, flow_ba))
            dice_A2B = self.cal_dice(
                A_seg_warped, B_seg)
            dice_B2A = self.cal_dice(
                B_seg_warped, A_seg)
            return dice_A2B, dice_B2A

    def save(self, pth_path, epoch):
        gen_path = os.path.join(pth_path, str(epoch)+"gen.pth")
        dis_path = os.path.join(pth_path, str(epoch)+"dis.pth")
        reg_path = os.path.join(pth_path, str(epoch)+"reg.pth")
        opt_path = os.path.join(pth_path, str(epoch)+"opt.pth")
        torch.save(
            {"a": self.gen_a2b.state_dict(), "b": self.gen_b2a.state_dict()}, gen_path
        )
        torch.save(
            {"a": self.dis_a2b.state_dict(), "b": self.dis_b2a.state_dict()}, dis_path
        )
        torch.save(self.reg.state_dict(), reg_path)
        torch.save(
            {
                "gen": self.gen_opt.state_dict(),
                "dis": self.dis_opt.state_dict(),
                "reg": self.reg_opt.state_dict(),
            },
            opt_path,
        )
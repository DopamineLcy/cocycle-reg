# system
import yaml

# torch
import torch
import torch.nn.functional as F
import torch.nn as nn

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)

class Transformer_3D(nn.Module):
    def __init__(self):
        super(Transformer_3D, self).__init__()

    def forward(self, src, flow):
        b = flow.shape[0]
        d = flow.shape[2]
        h = flow.shape[3]
        w = flow.shape[4]
        size = (d, h, w)
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = grid.to(torch.float32)
        grid = grid.repeat(b, 1, 1, 1, 1).cuda()
        new_locs = grid+flow
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * \
                (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2, 1, 0]]
        warped = F.grid_sample(
            src, new_locs, align_corners=True, padding_mode="border")

        return warped

def smooth_loss(flow):
    dy = torch.abs(flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :])
    dx = torch.abs(flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :])
    dz = torch.abs(flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1])

    dy = dy * dy
    dx = dx * dx
    dz = dz * dz

    d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
    grad = d / 3.0

    return grad

def save_result_log(epoch, iter, trainer, train_writer, cfg):
    # save log
    members = [attr for attr in dir(trainer)
               if not callable(getattr(trainer, attr)) and not attr.startswith("__") and ('loss' in attr)]
    for m in members:
        train_writer.add_scalar(m, getattr(trainer, m), epoch*90/cfg['batch_size']+iter)
    train_writer.flush()

def save_dice(epoch, dice_A2B, dice_B2A, train_writer):
    train_writer.add_scalar('dice_A2B', dice_A2B, epoch)
    train_writer.add_scalar('dice_B2A', dice_B2A, epoch)
    train_writer.flush()
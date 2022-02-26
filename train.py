# system
import os
import argparse
import tensorboardX
from tqdm import tqdm
# torch
import torch
from torch.utils.data import DataLoader
import shutil
# local
from data import ValDataset
from data import TrainDataset
from utils import save_dice
from utils import save_result_log
from utils import get_config
from trainer import Trainer


# config
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./config.yaml')
parser.add_argument("--local_rank", type=int, default=0)
opts = parser.parse_args()
cfg = get_config(opts.config)
os.environ["CUDA_VISIBLE_DEVICES"] = cfg['gpu_id']
torch.backends.cudnn.benchmark = True
train_root = cfg['in_path_train']
val_root = cfg['in_path_val']
out_path = cfg['out_path']
log_path = os.path.join(out_path, 'log/')
pth_path = os.path.join(out_path, 'pth/')
img_path = os.path.join(out_path, 'img/')
txt_path = os.path.join(out_path,'log.txt')
n_epoch = cfg['n_epoch']
batch_size = cfg['batch_size']

train_dataset = TrainDataset(train_root, cfg)
val_dataset = ValDataset(val_root, cfg)
# dataloader
dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
    persistent_workers=True
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=8,
    persistent_workers=True
)

train_writer = tensorboardX.SummaryWriter(log_path)
trainer = Trainer(cfg).cuda()
if __name__ == "__main__":
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    if not os.path.exists(pth_path):
        os.makedirs(pth_path)
    shutil.copy(opts.config, os.path.join(out_path, 'config.yaml'))
    # train begin
    start_epoch = 0
    best_dice = -1
    for epoch in range(start_epoch, n_epoch):
        # validation -----------------------
        trainer.gen_a2b.eval()
        trainer.gen_b2a.eval()
        trainer.dis_a2b.eval()
        trainer.dis_b2a.eval()
        trainer.reg.eval()
        dice_sum_A2B = 0
        dice_sum_B2A = 0
        dice_count = 0
        for modality1, modality2, seg_modality1, seg_modality2 in tqdm(val_dataloader):
            modality1 = modality1.cuda()
            modality2 = modality2.cuda()
            seg_modality1 = seg_modality1.cuda()
            seg_modality2 = seg_modality2.cuda()
            dice_A2B, dice_B2A = trainer.validate(modality1, modality2, seg_modality1, seg_modality2)
            if dice_A2B != -1:
                dice_sum_A2B += dice_A2B
                dice_count += 1
                dice_sum_B2A += dice_B2A
        dice_avg_A2B = dice_sum_A2B/dice_count
        dice_avg_B2A = dice_sum_B2A/dice_count
        save_dice(epoch, dice_avg_A2B, dice_avg_B2A, train_writer)
        if (dice_avg_A2B+dice_avg_B2A)/2>best_dice:
            best_dice = (dice_avg_A2B+dice_avg_B2A)/2
            trainer.save(pth_path, "best")
        trainer.save(pth_path,"newest")
        # validation end ---------------------
        trainer.gen_a2b.train()
        trainer.gen_b2a.train()
        trainer.dis_a2b.train()
        trainer.dis_b2a.train()
        trainer.reg.train()
        print("epochs:"+str(epoch))
        iter = 0
        for modality1, modality2 in tqdm(dataloader):
            modality1 = modality1.cuda()
            modality2 = modality2.cuda()
            trainer.set_input(modality1, modality2, epoch, iter, txt_path)
            trainer.optimize()
            save_result_log(epoch, iter, trainer,train_writer, cfg)
            iter += 1
        print("epoch:%03d/%03d" % (epoch+1, n_epoch))

        
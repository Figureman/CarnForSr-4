import os
import random
import numpy as np
import scipy.misc as misc
import skimage.measure as measure
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import TrainDataset, TestDataset
import time
from sample import sample

class Solver():
    def __init__(self, model, cfg):
        if cfg.scale > 0:
            self.refiner = model(scale=cfg.scale, 
                                 group=cfg.group)
        else:
            self.refiner = model(multi_scale=True, 
                                 group=cfg.group)
        
        #损失函数
        self.loss1 = nn.MSELoss()
        self.loss2 = nn.L1Loss()

        
        if cfg.loss_fn in ["MSE"]: 
            self.loss_fn = nn.MSELoss()
        elif cfg.loss_fn in ["L1"]: 
            self.loss_fn = nn.L1Loss()
        elif cfg.loss_fn in ["SmoothL1"]:
            self.loss_fn = nn.SmoothL1Loss()

        self.optim = optim.Adam(
            filter(lambda p: p.requires_grad, self.refiner.parameters()), 
            cfg.lr)
        
        self.train_data = TrainDataset(cfg.train_data_path, 
                                       scale=cfg.scale, 
                                       size=cfg.patch_size)
        self.train_loader = DataLoader(self.train_data,
                                       batch_size=cfg.batch_size,
                                       num_workers=1,
                                       shuffle=True, drop_last=True)
        
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.refiner = self.refiner.to(self.device)
        self.loss_fn = self.loss_fn
        self.test_data  = TestDataset("/workspace/SR/CARN-pytorch/dataset/DIV2K/DIV2K_valid", 4)
        self.cfg = cfg
        self.step = 0
        
        self.writer = SummaryWriter(log_dir=os.path.join("runs", cfg.ckpt_name))
        if cfg.verbose:
            num_params = 0
            for param in self.refiner.parameters():
                num_params += param.nelement()
            print("# of params:", num_params)

        os.makedirs(cfg.ckpt_dir, exist_ok=True)

    def fit(self):
        cfg = self.cfg
        refiner = nn.DataParallel(self.refiner, 
                                  device_ids=range(cfg.num_gpu))
        
        learning_rate = cfg.lr
        start_time =time.time()   #记录训练时间
        while True:
            for inputs in self.train_loader:
                self.refiner.train()

                if cfg.scale > 0:
                    scale = cfg.scale
                    hr, lr = inputs[-1][0], inputs[-1][1]
                else:
                    # only use one of multi-scale data
                    # i know this is stupid but just temporary
                    scale = random.randint(2, 4)
                    hr, lr = inputs[scale-2][0], inputs[scale-2][1]
                
                hr = hr.to(self.device)
                lr = lr.to(self.device)
                
                sr = refiner(lr, scale)
                #loss = self.loss_fn(sr, hr)
                alpha = 0.5
                beta = 0.5
                loss =alpha*self.loss1(sr,hr)+beta*self.loss2(sr,hr)
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(self.refiner.parameters(), cfg.clip)
                self.optim.step()

                learning_rate = self.decay_learning_rate()
                for param_group in self.optim.param_groups:
                    param_group["lr"] = learning_rate
                

                if   self.step % cfg.print_interval == 0:
                    if cfg.scale > 0:
                        elapsed_time =time.time()-start_time
                        #print("Step: {} ,Learning Rate: {:.6f} , Loss: {:.6f} , Elapsed Time: {:.2f} seconds".format(self.step,learning_rate,loss,elapsed_time))
                        psnr = self.sample(self.cfg)
                        print("Step: {} ,Learning Rate: {:.6f} , Loss: {:.6f} , Elapsed Time: {:.2f} seconds, Psnr in Testdata: {:.2f}".format(
                            self.step, learning_rate, loss, elapsed_time,psnr ))
                        
                    else:
                        print("Step: {} ,Learning Rate: {:.6f} , Loss: {:.6f}".format(self.stpe, learning_rate, loss))
                        psnr = [self.evaluate("dataset/Urban100", scale=i, num_step=self.step) for i in range(2, 5)]     
                    self.save(cfg.ckpt_dir, cfg.ckpt_name)
                self.step += 1
            if self.step > cfg.max_steps: break

    def load(self, path):
        self.refiner.load_state_dict(torch.load(path))
        splited = path.split(".")[0].split("_")[-1]
        try:
            self.step = int(path.split(".")[0].split("_")[-1])
        except ValueError:
            self.step = 0
        print("Load pretrained {} model".format(path))

    def save(self, ckpt_dir, ckpt_name):
        save_path = os.path.join(
            ckpt_dir, "{}_{}.pth".format(ckpt_name, self.step))
        torch.save(self.refiner.state_dict(), save_path)

    def decay_learning_rate(self):
        lr = self.cfg.lr * (0.5 ** (self.step // self.cfg.decay))
        return lr

    def sample(self, cfg):
        scale = cfg.scale
        device =self.device
        dataset=self.test_data
        self.refiner.eval()
        psnr = 0
        length = len(dataset)
        MaxValue =255
        for step, (hr, lr, name) in enumerate(dataset):
            if "DIV2K" in dataset.name:
                t1 = time.time()
                h, w = lr.size()[1:]
                h_half, w_half = int(h / 2), int(w / 2)
                h_chop, w_chop = h_half + cfg.shave, w_half + cfg.shave

                lr_patch = torch.zeros((4, 3, h_chop, w_chop), dtype=torch.float)
                lr_patch[0].copy_(lr[:, 0:h_chop, 0:w_chop])  # top left
                lr_patch[1].copy_(lr[:, 0:h_chop, w - w_chop:w])  # top right
                lr_patch[2].copy_(lr[:, h - h_chop:h, 0:w_chop])  # under left
                lr_patch[3].copy_(lr[:, h - h_chop:h, w - w_chop:w])  # under right
                lr_patch = lr_patch.to(device)

                sr = self.refiner(lr_patch, scale).detach()

                h, h_half, h_chop = h * scale, h_half * scale, h_chop * scale
                w, w_half, w_chop = w * scale, w_half * scale, w_chop * scale

                result = torch.zeros((3, h, w), dtype=torch.float).to(device)
                result[:, 0:h_half, 0:w_half].copy_(sr[0, :, 0:h_half, 0:w_half])
                result[:, 0:h_half, w_half:w].copy_(sr[1, :, 0:h_half, w_chop - w + w_half:w_chop])
                result[:, h_half:h, 0:w_half].copy_(sr[2, :, h_chop - h + h_half:h_chop, 0:w_half])
                result[:, h_half:h, w_half:w].copy_(sr[3, :, h_chop - h + h_half:h_chop, w_chop - w + w_half:w_chop])
                sr = result
                t2 = time.time()
            else:
                t1 = time.time()
                lr = lr.unsqueeze(0).to(device)
                sr = net(lr, cfg.scale).detach().squeeze(0)
                lr = lr.squeeze(0)
                t2 = time.time()
            sr =sr.cpu()
            sr1= (sr.numpy().astype(np.float64))
            hr1 = (hr.numpy().astype(np.float64))
            mse = np.mean((sr1-hr1)**2)
            psnrM = 10*np.log10((MaxValue**2)/mse)
            psnr += psnrM

        psnr = psnr / length
        return psnr

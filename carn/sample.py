import os
import json
import time
import importlib
import argparse
import cv2
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.autograd import Variable
from dataset import TestDataset
from PIL import Image
import torchvision.transforms as transforms
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,default="carn")
    parser.add_argument("--ckpt_path", type=str,default="/workspace/SR/CARN-pytorch/checkpoint/carn.pth")
    parser.add_argument("--group", type=int, default=1)
    parser.add_argument("--sample_dir", type=str ,default="/workspace/SR/CARN-pytorch/sample")
    parser.add_argument("--test_data_dir", type=str, default="/workspace/SR/CARN-pytorch/dataset/DIV2K/DIV2K_valid")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--shave", type=int, default=20)

    return parser.parse_args()


def save_image(tensor, filename):
    tensor = tensor.cpu()
    print(tensor.size())
    ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)


#Calculate Psnr
'''
>40db Excellent
30-40db It’s OK!
20-30db A little bad
<20db  Bad
'''
#计算两张图片的PSNR
def Psnr(img1,img2,MaxValue=255):

    # convert YUV space
    Img1 = cv2.imread(img1)
    Img1 = np.array(Img1,dtype=np.float64)

    Img2 = cv2.imread(img2)
    Img2 = np.array(Img2,dtype=np.float64)

    #MSE
    mse = np.mean((Img1-Img2)**2)

    return 10*np.log10((MaxValue**2)/mse)



#for one image
def lr2hr(net,device,imageDir, TargetDir,filename,scale=4,shave=20):

    transform = transforms.Compose([
            transforms.ToTensor()
    ])
    image = Image.open(imageDir).convert("RGB")
    imageT = transform(image)
    h,w = imageT.size()[1:]
    h_half, w_half = int(h / 2), int(w / 2)
    h_chop, w_chop = h_half + shave, w_half + shave


    lr_patch = torch.zeros((4, 3, h_chop, w_chop), dtype=torch.float)
    lr_patch[0].copy_(imageT[:, 0:h_chop, 0:w_chop])  # top left
    lr_patch[1].copy_(imageT[:, 0:h_chop, w - w_chop:w])  # top right
    lr_patch[2].copy_(imageT[:, h - h_chop:h, 0:w_chop])  # under left
    lr_patch[3].copy_(imageT[:, h - h_chop:h, w - w_chop:w])  # under right
    lr_patch = lr_patch.to(device)
    #Results
    sr = net(lr_patch, cfg.scale).detach()
    h, h_half, h_chop = h * scale, h_half * scale, h_chop * scale
    w, w_half, w_chop = w * scale, w_half * scale, w_chop * scale

    result = torch.zeros((3, h, w), dtype=torch.float).to(device)
    result[:, 0:h_half, 0:w_half].copy_(sr[0, :, 0:h_half, 0:w_half])
    result[:, 0:h_half, w_half:w].copy_(sr[1, :, 0:h_half, w_chop - w + w_half:w_chop])
    result[:, h_half:h, 0:w_half].copy_(sr[2, :, h_chop - h + h_half:h_chop, 0:w_half])
    result[:, h_half:h, w_half:w].copy_(sr[3, :, h_chop - h + h_half:h_chop, w_chop - w + w_half:w_chop])

    sr = result
    save_image(result[:, 0:h_half, 0:w_half],"/workspace/SR/CARN-pytorch/SaveImage/5.png")
    save_image(result[:, 0:h_half, w_half:w], "/workspace/SR/CARN-pytorch/SaveImage/6.png")
    save_image(result[:, h_half:h, 0:w_half], "/workspace/SR/CARN-pytorch/SaveImage/7.png")
    save_image(result[:, h_half:h, w_half:w], "/workspace/SR/CARN-pytorch/SaveImage/8.png")
    save_image(sr, "/workspace/SR/CARN-pytorch/SaveImage/finally.png")
    input()

    #Mkdir ResDir
    '''
    if not os.path.exits(TargetDir):
        os.makedirs(TargetDir)
    '''
    os.makedirs(TargetDir, exist_ok=True)
    print(TargetDir)
    TargetDir1=os.path.join(TargetDir,filename)
    print(TargetDir1)

    save_image(sr, TargetDir1)
def sample(net, device, dataset, cfg):
    scale = cfg.scale
    count=0
    Time =0
    psnr = 0
    length = len(dataset)
    for step, (hr, lr, name) in enumerate(dataset):
        if "DIV2K" in dataset.name:
            t1 = time.time()
            h, w = lr.size()[1:]
            h_half, w_half = int(h/2), int(w/2)
            h_chop, w_chop = h_half + cfg.shave, w_half + cfg.shave

            lr_patch = torch.zeros((4, 3, h_chop, w_chop), dtype=torch.float)
            lr_patch[0].copy_(lr[:, 0:h_chop, 0:w_chop])  # top left
            lr_patch[1].copy_(lr[:, 0:h_chop, w-w_chop:w]) #top right
            lr_patch[2].copy_(lr[:, h-h_chop:h, 0:w_chop]) #under left
            lr_patch[3].copy_(lr[:, h-h_chop:h, w-w_chop:w]) #under right
            lr_patch = lr_patch.to(device)
            
            sr = net(lr_patch, cfg.scale).detach()
            
            h, h_half, h_chop = h*scale, h_half*scale, h_chop*scale
            w, w_half, w_chop = w*scale, w_half*scale, w_chop*scale

            result = torch.zeros((3, h, w), dtype=torch.float).to(device)
            result[:, 0:h_half, 0:w_half].copy_(sr[0, :, 0:h_half, 0:w_half])
            result[:, 0:h_half, w_half:w].copy_(sr[1, :, 0:h_half, w_chop-w+w_half:w_chop])
            result[:, h_half:h, 0:w_half].copy_(sr[2, :, h_chop-h+h_half:h_chop, 0:w_half])
            result[:, h_half:h, w_half:w].copy_(sr[3, :, h_chop-h+h_half:h_chop, w_chop-w+w_half:w_chop])
            sr = result
            t2 = time.time()
        else:
            t1 = time.time()
            lr = lr.unsqueeze(0).to(device)
            sr = net(lr, cfg.scale).detach().squeeze(0)
            lr = lr.squeeze(0)
            t2 = time.time()

        model_name = cfg.ckpt_path.split(".")[0].split("/")[-1]
        sr_dir = os.path.join(cfg.sample_dir,
                              model_name, 
                              cfg.test_data_dir.split("/")[-1],
                              "x{}".format(cfg.scale),
                              "SR")
        hr_dir = os.path.join(cfg.sample_dir,
                              model_name, 
                              cfg.test_data_dir.split("/")[-1],
                              "x{}".format(cfg.scale),
                              "HR")
        
        os.makedirs(sr_dir, exist_ok=True)
        os.makedirs(hr_dir, exist_ok=True)
        nameN =name[0:4]+"x4"+name[4:8]
        sr_im_path = os.path.join(sr_dir,"{}".format(nameN.replace("HR", "SR")))
        hr_im_path = os.path.join(hr_dir, "{}".format(name))

        save_image(sr, sr_im_path) #*4 Results
        save_image(hr, hr_im_path) # Original

        psnrN = Psnr(sr_im_path, hr_im_path, MaxValue=255)

        print("Saved {} ({}x{} -> {}x{}, {:.3f}s, {:.3f})"
            .format(sr_im_path, lr.shape[1], lr.shape[2], sr.shape[1], sr.shape[2], t2-t1 , psnrN))
        str = "Saved {} ({}x{} -> {}x{}, {:.3f}s, {:.3f})".format(sr_im_path, lr.shape[1], lr.shape[2], sr.shape[1], sr.shape[2], t2-t1 , psnrN)
        os.makedirs("/workspace/SR/CARN-pytorch/log/loger1.txt", exist_ok=True)
        with open("/workspace/SR/CARN-pytorch/log/loger.txt",'a') as f:
            f.write(str)
            f.write('\n')
        psnr += psnrN
        Time+=(t2-t1)
    Time = Time / length
    print(Time)
    psnr = psnr / length
    print(psnr)

def sampleT(net, device, cfg, dir,targetPath,scale=4,shave=20):
    filenames = os.listdir(dir)
    filenames = sorted(filenames)
    transform = transforms.Compose([
            transforms.ToTensor()
    ])
    Time = 0
    length = len(filenames)
    #print(filenames)
    for i in filenames:
        t1 = time.time()
        imagePath = os.path.join(dir,i)
        #print(imagePath)
        resPath = os.path.join(targetPath,i)
        #print(resPath)
        imageT = transform(Image.open(imagePath).convert("RGB"))
        #print(imageT)
        h, w = imageT.size()[1:]
        h_half, w_half = int(h / 2), int(w / 2)
        h_chop, w_chop = h_half + shave, w_half + shave

        lr_patch = torch.zeros((4, 3, h_chop, w_chop), dtype=torch.float)
        lr_patch[0].copy_(imageT[:, 0:h_chop, 0:w_chop])  # top left
        lr_patch[1].copy_(imageT[:, 0:h_chop, w - w_chop:w])  # top right
        lr_patch[2].copy_(imageT[:, h - h_chop:h, 0:w_chop])  # under left
        lr_patch[3].copy_(imageT[:, h - h_chop:h, w - w_chop:w])  # under right
        lr_patch = lr_patch.to(device)
        # Results
        sr = net(lr_patch, cfg.scale).detach()
        h, h_half, h_chop = h * scale, h_half * scale, h_chop * scale
        w, w_half, w_chop = w * scale, w_half * scale, w_chop * scale

        result = torch.zeros((3, h, w), dtype=torch.float).to(device)
        result[:, 0:h_half, 0:w_half].copy_(sr[0, :, 0:h_half, 0:w_half])
        result[:, 0:h_half, w_half:w].copy_(sr[1, :, 0:h_half, w_chop - w + w_half:w_chop])
        result[:, h_half:h, 0:w_half].copy_(sr[2, :, h_chop - h + h_half:h_chop, 0:w_half])
        result[:, h_half:h, w_half:w].copy_(sr[3, :, h_chop - h + h_half:h_chop, w_chop - w + w_half:w_chop])
        sr = result
        save_image(sr, resPath)
        t2 = time.time()
        print("Saved {} ( {:.3f}s)"
              .format(resPath, t2 - t1))
        Time =Time +t2-t1
    print(Time/length)
def main(cfg):
    module = importlib.import_module("model.{}".format(cfg.model))
    net = module.Net(multi_scale=True, 
                     group=cfg.group)
    print(json.dumps(vars(cfg), indent=4, sort_keys=True))
    state_dict = torch.load(cfg.ckpt_path)
    '''
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        # name = k[7:] # remove "module."
        new_state_dict[name] = v
    '''
    net.load_state_dict(state_dict,strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    dir = "/workspace/SR/DIV2K/DIV2K_test_LR_bicubic/X4"
    targetPath = "/workspace/SR/DIV2K/result"
    #测试结果（上传到官方测试）
    sampleT(net, device, cfg, dir,targetPath)
    #lr2hr(net, device,"/workspace/SR/DIV2K/DIV2K_valid/DIV2K_valid_LR_bicubic/X4/0801x4.png", "/workspace/SR/DIV2K/temp/","a.png")
    #dataset = TestDataset(cfg.test_data_dir, cfg.scale)

    #sample(net, device, dataset, cfg)

if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
    #a=Psnr("/workspace/SR/CARN-pytorch/sample/carn/DIV2K_valid/x4/HR/0801.png","/workspace/SR/CARN-pytorch/sample/carn/DIV2K_valid/x4/SR/0801.png")
    #PSNR计算
    #print(a)

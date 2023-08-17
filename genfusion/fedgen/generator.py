import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torchvision.utils as utils
from fasgan.generator import Generator
from fasgan.discriminator import Discriminator
import numpy as np
from tqdm import tqdm
from PIL import Image

class GeneratorFramework():

    def __init__(self, dataroot=None, difference=None, worker=None, category=None):
        self.dataroot = dataroot
        self.category = category
        self.difference = difference
        self.worker = worker
        self.workers = 2
        self.batch_size = 48
        self.image_size = 64
        self.n_channels = 3
        self.nz = 100
        self.gen_dim = 64
        self.dis_dim = 64
        self.num_epochs = 5
        self.lr = 0.005
        self.beta1 = 0.3
        self.beta2 = 0.9
        self.ngpu = 1
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu>0) else 'cpu')
        self.criterion = nn.BCELoss()
        self.fixed_noise = torch.randn(self.image_size, self.nz, 1, 1, device=self.device)
        self.real_label = 1.
        self.fake_label = 0.

    def create_loader(self):
        dataset = dataset.ImageFolder(root=self.dataroot,
            transform=transforms.Compose([
            transforms.Resize(100),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ]))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def define_raw_blocks(self):

        netG = Generator().to(self.device)
        if (self.device.type == 'cuda') and (self.ngpu > 1):
            netG = nn.DataParallel(netG, list(range(self.ngpu)))
        netG.apply(self.weights_init)

        netD = Discriminator().to(self.device)

        if (self.device.type == 'cuda') and (self.ngpu > 1):
            netD = nn.DataParallel(netD, list(range(self.ngpu)))
        netD.apply(self.weights_init)

        optimizerG = optim.Adam(netG.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        optimizerD = optim.Adam(netD.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

        return netG, netD, optimizerG, optimizerD

    def train(self, dataloader, netD, netG, optimizerD, optimizerG):
        img_list = list()
        G_losses = list()
        D_losses = list()
        iters = 0

        print("Starting Training Loop...")

        for epoch in tqdm(range(self.num_epochs)):
            
            for i, data in enumerate(dataloader, 0):
                
                netD.zero_grad()
                real_cpu = data[0].to(self.device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), self.real_label, dtype=torch.float, device=self.device)
                
                output = netD(real_cpu).view(-1)
                errD_real = self.criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()
                
                noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
                fake = netG(noise)
                label.fill_(self.fake_label)
                
                output = netD(fake.detach()).view(-1)
                errD_fake = self.criterion(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                
                errD = errD_real + errD_fake
                optimizerD.step()
                
                netG.zero_grad()
                label.fill_(self.real_label)
                output = netD(fake).view(-1)
                errG = self.criterion(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                optimizerG.step()
                
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                        % (epoch, self.num_epochs, i, len(dataloader),
                            errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                    
                G_losses.append(errG.item())
                D_losses.append(errD.item())
                
                # if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                #     with torch.no_grad():
                #         fake = netG(fixed_noise).detach().cpu()
                #     img_list.append(utils.make_grid(fake, padding=2, normalize=True))
                    
                iters += 1

        return netG, netD

    def generate_data(self, netG, netD):
        for idx in tqdm(range(len(self.difference))):
            with torch.no_grad():
                noise = torch.randn(self.image_size, self.nz, 1, 1, device=self.device)
                fake = netG(noise)
                output = netD(fake.detach()).view(-1)
                output = np.array(output.to('cpu'))
                max_conf = max(output)
                index = output.index(max_conf)
                datapoint = transforms.ToPILImage()(fake[index])
                datapoint.save(f'/home/ubuntu/GenAI-Rush/generated/{self.worker}/{self.category}/{self.category}_generated_{idx}.jpg')

    def process(self):
        dataloader = self.create_loader()
        netG, netD, optimizerG, optimizerD = self.define_raw_blocks()
        netG, netD = self.train(dataloader, netD, netG, optimizerD, optimizerG)
        generate_data(netG, netD)
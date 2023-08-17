import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch.autograd import Variable
import logging
import pandas as pd
import numpy as np
import os
import torch.nn as nn
from fedgen.utils import PrivateModelBuilder
from fedgen.dataset import FedDataset, transform
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import models

criterion = nn.CrossEntropyLoss()

class FedTrainer():

    def __init__(self, worker=None, categories=None, batch_size=None, epochs=None, 
        MAX_PHYSICAL_BATCH_SIZE=None, MAX_GRAD_NORM=None, EPSILON=None, DELTA=None, trainDataLoader=None, valDataLoader=None):
        self.worker = worker
        self.categories = categories
        self.batch_size = batch_size
        self.epochs = epochs
        self.MAX_PHYSICAL_BATCH_SIZE = MAX_PHYSICAL_BATCH_SIZE
        self.MAX_GRAD_NORM = MAX_GRAD_NORM
        self.EPSILON = EPSILON
        self.DELTA = DELTA
        self.val_loader = valDataLoader
        self.baseModel = models.resnet101(pretrained=False)
        num_features = self.baseModel.fc.in_features
        self.baseModel.fc = nn.Linear(num_features, self.categories)
        modelBuilder = PrivateModelBuilder(model=self.baseModel, trainDataLoader=trainDataLoader)
        self.encryptedModel, self.optimizer, self.train_loader, self.privacy_engine = modelBuilder.privatization(MAX_GRAD_NORM=self.MAX_GRAD_NORM,
                                                                                            EPSILON=self.EPSILON, DELTA=self.DELTA, EPOCHS=self.epochs)
        self.encryptedModel.load_state_dict(torch.load(f'/home/ubuntu/GenAI-Rush/models/encrypted_{self.worker}.pth'), strict=False)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1, verbose=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fetchLossAcc(self, output, labels):
        loss = criterion(output, labels)
        pred = output.detach().cpu().numpy()
        pred = [np.argmax(pred[idx]) for idx in range(len(pred))]
        labels = labels.detach().cpu().numpy()
        correct = [True for i in range(len(labels)) if labels[i]==pred[i]]
        acc = len(correct)/len(labels)

        return loss, acc, labels, pred

    def runTrainingSession(self):

        tb = SummaryWriter()

        for epoch in range(self.epochs):
    
            print(f'EPOCH {epoch+1}:\n')
            
            train_loss = 0.0
            val_loss = 0.0
            train_acc = 0.0
            val_acc = 0.0
            rounds = 0
            
            self.encryptedModel = self.encryptedModel.to(self.device)
            self.encryptedModel.train()
            
            with BatchMemoryManager(data_loader=self.train_loader, 
                                    max_physical_batch_size=self.MAX_PHYSICAL_BATCH_SIZE, 
                                    optimizer=self.optimizer) as memory_safe_data_loader:
                
                for i, (images, labels) in enumerate(memory_safe_data_loader):
                
                    assert len(images) <= self.MAX_PHYSICAL_BATCH_SIZE

                    self.optimizer.zero_grad()
                    images = Variable(images).to(device=self.device)
                    labels = Variable(labels).to(device=self.device)
                    grid = torchvision.utils.make_grid(images)
                    tb.add_image(f"images_{self.worker}", grid)

                    output = self.encryptedModel(images)
                    loss, acc, _, _ = self.fetchLossAcc(output, labels)

                    train_loss += loss.item()
                    train_acc += acc
                    loss.backward()
                    self.optimizer.step()
                    
                    rounds += 1

                    torch.cuda.empty_cache()

                    tb.add_scalar(f"Training Loss: {self.worker}", loss.item(), epoch+1)
                    tb.add_scalar(f"Training Accuracy: {self.worker}", acc, epoch+1)


            train_loss /= rounds
            train_acc /= rounds
            logging.info('Training Round complete, Calculating Epsilon & Delta...')

            logging.info('Starting Validation...')
            self.encryptedModel.eval()
            rounds = 0
            
            for i, (images, labels) in enumerate(self.val_loader):
                
                images = Variable(images).to(device=self.device)
                labels = Variable(labels).to(device=self.device)
                output = self.encryptedModel(images)
                loss, acc, _, _ = self.fetchLossAcc(output, labels)
                
                val_loss += loss.item()
                val_acc += acc
                
                rounds += 1

                tb.add_scalar(f"Validation Loss: {self.worker}", loss.item(), epoch+1)
                tb.add_scalar(f"Validation Accuracy: {self.worker}", acc, epoch+1)
                
            val_loss /= rounds
            val_acc /= rounds

            self.scheduler.step()
            epsilon = self.privacy_engine.get_epsilon(self.DELTA)
            
            performance = {
                'Epoch': epoch+1,
                'Loss': round(train_loss, 4),
                'Val Loss': round(val_loss, 4),
                'Train Acc': round(train_acc, 4),
                'Val Acc': round(val_acc, 4),
                'ε': round(epsilon, 4),
                'δ': self.DELTA
            }
            logging.info(performance, '\n')
            torch.cuda.empty_cache()
        
        self.encryptedModel.eval()
        self.encryptedModel.to(device='cpu')
        torch.save(self.encryptedModel.state_dict(), f'/home/ubuntu/GenAI-Rush/models/encrypted_{self.worker}.pth')
        torch.cuda.empty_cache()

        tb.close()

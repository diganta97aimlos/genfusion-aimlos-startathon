import os
import torch
from torchvision import models
import torch.nn as nn
from torch import optim
from fedgen.utils import PrivateModelBuilder
import logging

class SimpleFederatedAveraging():

    def __init__(self, workers=None, categories=None, MAX_GRAD_NORM=None, EPSILON=None, DELTA=None, EPOCHS=None):
        self.workers=workers
        self.categories = categories
        self.baseModel = models.resnet101(pretrained=False)
        num_features = self.baseModel.fc.in_features
        self.baseModel.fc = nn.Linear(num_features, self.categories)
        self.MAX_GRAD_NORM = MAX_GRAD_NORM
        self.EPSILON = EPSILON
        self.DELTA = DELTA
        self.EPOCHS = EPOCHS

    def loadModels(self, trainDataLoader=None, worker=None):
        modelBuilder = PrivateModelBuilder(model=self.baseModel, trainDataLoader=trainDataLoader)
        encryptedModel, _, _, _ = modelBuilder.privatization(MAX_GRAD_NORM=self.MAX_GRAD_NORM, EPSILON=self.EPSILON, DELTA=self.DELTA, EPOCHS=self.EPOCHS)
        encryptedModel.load_state_dict(torch.load(f'/home/ubuntu/GenAI-Rush/models/encrypted_{worker}.pth'))
        return encryptedModel

    def federateGlobalModel(self, loaders=None):
        logging.info(f'Federating {len(self.workers)} Client Models')
        clientModelsWeights = list()
        for idx in range(len(self.workers)):
            clientModelsWeights.append(self.loadModels(trainDataLoader=loaders[idx], worker=self.workers[idx]).state_dict())
        self.federatedModel = self.loadModels(trainDataLoader=loaders[0], worker='worker1')
        for key in self.federatedModel.state_dict():
            for index in range(len(self.workers)):
                if index==0:
                    self.federatedModel.state_dict()[key] = clientModelsWeights[index][key]
                else:
                    self.federatedModel.state_dict()[key] +=  clientModelsWeights[index][key]
        for key in self.federatedModel.state_dict():
            self.federatedModel.state_dict()[key] /= len(self.workers)
        torch.save(self.federatedModel.state_dict(), '/home/ubuntu/GenAI-Rush/models/encrypted_global_federated.pth')

    def federateClientModels(self):
        clientModelsWeights = list()
        for worker in self.workers:
            clientModelsWeights.append(torch.load(f'/home/ubuntu/GenAI-Rush/models/encrypted_{worker}.pth'))
        federateGlobalModelWeights = torch.load('/home/ubuntu/GenAI-Rush/models/encrypted_global_federated.pth')
        for idx in range(len(clientModelsWeights)):
            worker = self.workers[idx]
            for key in federateGlobalModelWeights:
                clientModelsWeights[idx][key] = (0.75*clientModelsWeights[idx][key]) + (0.25*federateGlobalModelWeights[key])
            torch.save(clientModelsWeights[idx], f'/home/ubuntu/GenAI-Rush/models/encrypted_federated_{worker}.pth')
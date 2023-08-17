import os
import torch
from torch import nn, optim
from fedgen.utils import PrivateModelBuilder
from fedgen.testModule import TestingModule
from fedgen.utils import CreateLoaders
from torchvision import models
import pandas as pd

class TestFederated():

    def __init__(self, workers=None, categories=None, MAX_GRAD_NORM=None, EPSILON=None, DELTA=None, EPOCHS=None, max_physical_batch_size=None, 
                 save_path=None, batch_size=None):
        self.workers=workers
        self.categories = categories
        self.baseModel = models.resnet18(pretrained=False)
        num_features = self.baseModel.fc.in_features
        self.baseModel.fc = nn.Linear(num_features, self.categories)
        self.MAX_GRAD_NORM = MAX_GRAD_NORM
        self.EPSILON = EPSILON
        self.DELTA = DELTA
        self.EPOCHS = EPOCHS
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_physical_batch_size = max_physical_batch_size
        self.batch_size = batch_size
        self.save_path = save_path
        self.performances = {'Client': list(), 'Federated': list(), 'Test Loss': list(), 'Test Accuracy': list(), 
            'Precision': list(), 'Recall': list(), 'F1 Score': list()}

    def loadModels(self, trainDataLoader=None, weightsPath=None):
        modelBuilder = PrivateModelBuilder(model=self.baseModel, trainDataLoader=trainDataLoader)
        encryptedModel, _, _, _ = modelBuilder.privatization(MAX_GRAD_NORM=self.MAX_GRAD_NORM, EPSILON=self.EPSILON, DELTA=self.DELTA, EPOCHS=self.EPOCHS)
        encryptedModel.load_state_dict(torch.load(weightsPath))
        return encryptedModel

    def testClientModels(self):
        testObj = TestingModule(device=self.device)
        for worker in self.workers:
            loaderObj = CreateLoaders(save_path=self.save_path, batch_size=self.batch_size, max_physical_batch_size=self.max_physical_batch_size, worker=worker)
            trainDataLoader, _, testDataLoader = loaderObj.fetchLoaders()
            clientModel = self.loadModels(trainDataLoader=trainDataLoader, weightsPath=f'/home/ubuntu/GenAI-Rush/models/encrypted_{worker}.pth')
            performance = testObj.runTestSession(model=clientModel, testDataLoader=testDataLoader)
            self.performances['Client'].append(worker.capitalize())
            self.performances['Federated'].append(False)
            self.performances['Test Loss'].append(performance['test_loss'])
            self.performances['Test Accuracy'].append(performance['test_acc'])
            self.performances['Precision'].append(performance['precision'])
            self.performances['Recall'].append(performance['recall'])
            self.performances['F1 Score'].append(performance['f1_score'])

    def testFederatedClientModels(self):
        testObj = TestingModule(device=self.device)
        for worker in self.workers:
            loaderObj = CreateLoaders(save_path=self.save_path, batch_size=self.batch_size, max_physical_batch_size=self.max_physical_batch_size, worker=worker)
            trainDataLoader, _, testDataLoader = loaderObj.fetchLoaders()
            clientModel = self.loadModels(trainDataLoader=trainDataLoader, weightsPath=f'/home/ubuntu/GenAI-Rush/models/encrypted_{worker}.pth')
            performance = testObj.runTestSession(model=clientModel, testDataLoader=testDataLoader)
            self.performances['Client'].append(worker.capitalize())
            self.performances['Federated'].append(True)
            self.performances['Test Loss'].append(performance['test_loss'])
            self.performances['Test Accuracy'].append(performance['test_acc'])
            self.performances['Precision'].append(performance['precision'])
            self.performances['Recall'].append(performance['recall'])
            self.performances['F1 Score'].append(performance['f1_score'])

    def testFederatedGlobalModel(self):
        testObj = TestingModule(device=self.device)
        for worker in self.workers:
            loaderObj = CreateLoaders(save_path=self.save_path, batch_size=self.batch_size, max_physical_batch_size=self.max_physical_batch_size, worker=worker)
            trainDataLoader, _, testDataLoader = loaderObj.fetchLoaders()
            fedGlobalModel = self.loadModels(trainDataLoader=trainDataLoader, weightsPath='/home/ubuntu/GenAI-Rush/models/encrypted_global_federated.pth')
            performance = testObj.runTestSession(model=fedGlobalModel, testDataLoader=testDataLoader)
            self.performances['Client'].append(f'Global - {worker.capitalize()}')
            self.performances['Federated'].append(True)
            self.performances['Test Loss'].append(performance['test_loss'])
            self.performances['Test Accuracy'].append(performance['test_acc'])
            self.performances['Precision'].append(performance['precision'])
            self.performances['Recall'].append(performance['recall'])
            self.performances['F1 Score'].append(performance['f1_score'])
        return pd.DataFrame(self.performances)

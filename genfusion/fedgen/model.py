import torch, os
from fedgen.dataset import FedDataset
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.utils.batch_memory_manager import BatchMemoryManager
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
import pickle
from fedgen.utils import PrivateModelBuilder
from torchvision import models

class EncryptData():

    def __init__(self, categories=None, worker=None):
        self.categories = categories
        self.model = models.resnet101(pretrained=False)
        num_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_features, self.categories)
        self.worker = worker

    def encrypt(self, trainDataLoader=None, epochs=None, MAX_GRAD_NORM=None, EPSILON=None, DELTA=None):
        modelBuilder = PrivateModelBuilder(model=self.model, trainDataLoader=trainDataLoader)
        self.model, _, _, _ = modelBuilder.privatization(MAX_GRAD_NORM=MAX_GRAD_NORM, EPSILON=EPSILON, DELTA=DELTA, EPOCHS=epochs)
        torch.save(self.model.state_dict(), f'/home/ubuntu/GenAI-Rush/models/encrypted_{self.worker}.pth')

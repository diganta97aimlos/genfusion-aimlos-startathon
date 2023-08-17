from opacus.optimizers import DPOptimizer
from opacus.validators import ModuleValidator
from opacus import GradSampleModule
from opacus.data_loader import DPDataLoader
from opacus.accountants import RDPAccountant
import torch
from fedgen.dataset import FedDataset, transform, test_transform
from torch.utils.data import DataLoader
from opacus.privacy_engine import PrivacyEngine
import pandas as pd
import os

class PrivateModelBuilder():

    def __init__(self, model=None, trainDataLoader=None):
        self.model=model
        self.trainDataLoader=trainDataLoader

    def privatization(self, MAX_GRAD_NORM=None, EPSILON=None, DELTA=None, EPOCHS=None):
        errors = ModuleValidator.validate(self.model, strict=False)
        if len(errors) > 0:
            self.model = ModuleValidator.fix(self.model)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        # self.model = GradSampleModule(self.model)
        privacy_engine = PrivacyEngine()
        self.model, self.optimizer, self.train_loader = privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.trainDataLoader,
            epochs=EPOCHS,
            target_epsilon=EPSILON,
            target_delta=DELTA,
            max_grad_norm=MAX_GRAD_NORM,
        )
        # self.train_loader = DPDataLoader.from_data_loader(self.trainDataLoader, distributed=False)
        # sample_rate = 1 / len(self.train_loader)
        # expected_batch_size = int(len(self.train_loader.dataset) * sample_rate)
        # self.optimizer = DPOptimizer(
        #     optimizer=self.optimizer,
        #     noise_multiplier=1.0,
        #     max_grad_norm=MAX_GRAD_NORM,
        #     expected_batch_size=expected_batch_size,
        # )
        # accountant = RDPAccountant()
        # self.optimizer.attach_step_hook(accountant.get_optimizer_hook_fn(sample_rate=sample_rate))
        return self.model, self.optimizer, self.train_loader, privacy_engine

class CreateLoaders():

    def __init__(self, save_path=None, batch_size=None, max_physical_batch_size=None, worker=None):
        self.save_path = save_path
        self.batch_size = batch_size
        self.max_physical_batch_size = max_physical_batch_size
        self.worker = worker

    def fetchLoaders(self):
        trainDf = pd.read_csv(os.path.join(self.save_path, f'train_{self.worker}.csv'), index_col=False)
        train_file_paths = trainDf.file_path
        trainDataset = FedDataset(file_paths=train_file_paths, labels=trainDf.category.tolist(), transform=transform)
        trainDataLoader = DataLoader(trainDataset, batch_size=self.batch_size, shuffle=True)

        valDf = pd.read_csv(os.path.join(self.save_path, f'val_{self.worker}.csv'), index_col=False)
        val_file_paths = valDf.file_path
        valDataset = FedDataset(file_paths=val_file_paths, labels=valDf.category.tolist(), transform=transform)
        valDataLoader = DataLoader(valDataset, batch_size=self.max_physical_batch_size, shuffle=True)

        testDf = pd.read_csv(os.path.join(self.save_path, f'test_{self.worker}.csv'), index_col=False)
        test_file_paths = testDf.file_path
        testDataset = FedDataset(file_paths=test_file_paths, labels=testDf.category.tolist(), transform=test_transform)
        testDataLoader = DataLoader(testDataset, batch_size=self.max_physical_batch_size, shuffle=True)        

        return trainDataLoader, valDataLoader, testDataLoader

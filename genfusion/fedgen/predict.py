import torch
import os
from fas14mnet import Fas14MNet
from torchvision import models
from torch import nn, optim
from packages.fedgen.fedgen.dataset import PredictionDataset, pred_transform
from packages.fedgen.fedgen.utils import PrivateModelBuilder
from packages.fedgen.fedgen.utils import CreateLoaders
from torch.utils.data import DataLoader
import cv2
import pandas as pd
from torch.autograd import Variable
import numpy as np

class Predict():

    def __init__(self, client=None, num_categories=None, mode=None, MAX_GRAD_NORM=None,
                 EPSILON=None, DELTA=None, EPOCHS=None, batch_size=None, modelSelected=None, train_loader=None):
        self.client = client
        self.num_categories = num_categories
        self.mode = mode
        self.MAX_GRAD_NORM = MAX_GRAD_NORM
        self.EPSILON = EPSILON
        self.DELTA = DELTA
        self.EPOCHS = EPOCHS
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.modelSelected = modelSelected
        self.model = models.resnet18(pretrained=False)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.num_categories)
        self.train_loader = train_loader
        self.image_path = '/home/ubuntu/kreedaAI/fedgen/to_predict'
        self.mapping = pd.read_csv(f'/home/ubuntu/kreedaAI/fedgen/workers/train_{self.client}.csv', index_col=False)

    def loadModel(self):
        if self.modelSelected == 'Client Model - Encrypted':
            selectedWeights = f'encrypted_{self.client}.pth'
        elif self.modelSelected == 'Client Model - Encrypted & Federated':
            selectedWeights = f'encrypted_federated_{self.client}.pth'
        else:
            selectedWeights = f'encrypted_global_federated.pth'
        modelBuilder = PrivateModelBuilder(model=self.model, trainDataLoader=self.train_loader)
        self.model, _, _, _ = modelBuilder.privatization(MAX_GRAD_NORM=self.MAX_GRAD_NORM, EPSILON=self.EPSILON, DELTA=self.DELTA, EPOCHS=self.EPOCHS)
        self.model.load_state_dict(torch.load(f'/home/ubuntu/kreedaAI/fedgen/models/{selectedWeights}'))
        self.model.eval()
        self.model.to(self.device)

    def create_pred_loader(self, file_paths=None):
        predDataset = PredictionDataset(file_paths=file_paths, transform=pred_transform)
        predDataLoader = DataLoader(predDataset, batch_size=self.batch_size, shuffle=False)
        return predDataLoader

    def predict_single_files(self):
        files = os.listdir(self.image_path)
        files = [f for f in files if f.endswith('jpg')]
        predictions = list()
        for index in range(len(files)):
            filePath = os.path.join(self.image_path, files[index])
            image = cv2.imread(filePath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            aug = pred_transform(image=image)
            image = aug['image']
            image = image.unsqueeze(0)
            image = image.to(self.device)
            output = self.model(image)
            pred = output.detach().cpu().numpy()
            pred = [np.argmax(pred[idx]) for idx in range(len(pred))][0]
            prediction = self.mapping[self.mapping.category==pred]['file_path'].tolist()[0].split('/')[-2]
            predictions.append(prediction)
        return predictions, files

    def predict_fileset(self):
        files = os.listdir(self.image_path)
        zipFile = [f for f in files if f.endswith('zip')][0]
        directory = zipFile.split('.')[0]
        file_paths = os.listdir(f'{self.image_path}/{directory}')
        file_paths = [f'{self.image_path}/{directory}/{path}' for path in file_paths]
        prediction_loader = self.create_pred_loader(file_paths=file_paths)
        predictions = list()
        for i, images in enumerate(prediction_loader):
            images = Variable(images).to(self.device)
            output = self.model(images)
            pred = output.detach().cpu().numpy()
            pred = [np.argmax(pred[idx]) for idx in range(len(pred))]
            for element in pred:
                prediction = self.mapping[self.mapping.category==element]['file_path'].tolist()[0].split('/')[-2]
                predictions.append(prediction)
        return predictions, file_paths

    def run(self):
        self.loadModel()
        if self.mode == 'Single Files (Max 5)':
            predictions, files = self.predict_single_files()
            files = [f'/home/ubuntu/kreedaAI/fedgen/to_predict/{file}' for file in files]
            result_dict = {'File Path': list(), 'Prediction': list()}
            for idx in range(len(files)):
                result_dict['File Path'].append(files[idx])
                result_dict['Prediction'].append(predictions[idx])
        else:
            predictions, file_paths = self.predict_fileset()
            result_dict = {'File Path': list(), 'Prediction': list()}
            for idx in range(len(file_paths)):
                result_dict['File Path'].append(file_paths[idx])
                result_dict['Prediction'].append(predictions[idx])
        results = pd.DataFrame(result_dict)
        return results

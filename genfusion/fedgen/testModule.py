import os
import torch
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import precision_score, recall_score, f1_score
import torch.nn as nn
import logging

criterion = nn.CrossEntropyLoss()

class TestingModule():

    def __init__(self, device=None):
        self.device = device

    def fetchLossAcc(self, output, labels):
        loss = criterion(output, labels)
        pred = output.detach().cpu().numpy()
        pred = [np.argmax(pred[idx]) for idx in range(len(pred))]
        labels = labels.detach().cpu().numpy()
        correct = [True for i in range(len(labels)) if labels[i]==pred[i]]
        acc = len(correct)/len(labels)

        return loss, acc, labels, pred

    def runTestSession(self, model=None, testDataLoader=None):

        test_loss = 0.0
        test_acc = 0.0
        precision = 0.0
        recall = 0.0
        F1_score = 0.0
        counter = 0

        model.eval()
        model.to(self.device)

        with torch.no_grad():

            for i, (images, labels) in enumerate(testDataLoader):

                images = Variable(images).to(self.device)
                labels = Variable(labels).to(self.device)

                output = model(images)

                loss, acc, true, predictions = self.fetchLossAcc(output, labels)

                test_loss += loss.item()
                test_acc += acc

                precision += precision_score(true, predictions, average='weighted')
                recall += recall_score(true, predictions, average='weighted')
                F1_score += f1_score(true, predictions, average='weighted')

                counter += 1

                torch.cuda.empty_cache()

        precision /= counter
        recall /= counter
        F1_score /= counter
        test_loss /= counter
        test_acc /= counter

        performance = {'test_loss': round(test_loss, 4), 'test_acc': round(test_acc, 4), 
            'precision': round(precision, 4), 'recall': round(recall, 4), 'f1_score': round(F1_score, 4)}

        torch.cuda.empty_cache()

        return performance
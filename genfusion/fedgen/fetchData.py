import os
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

class PreProcess():

    def __init__(self, data_dir=None, worker=None, save_path=None):
        self.save_path = save_path
        self.worker = worker
        if data_dir is not None and worker is not None:
            self.data_dir = os.path.join(data_dir, self.worker)
            self.categories = os.listdir(self.data_dir)

    def splitData(self):
        data = {'imageId': list(), 'category': list(), 'file_path': list()}
        for category in self.categories:
            for filename in os.listdir(f'{self.data_dir}/{category}'):
                imageId = filename.split('.')[0]
                file_path = os.path.join(self.data_dir, category, filename)
                data['imageId'].append(imageId)
                data['category'].append(category)
                data['file_path'].append(file_path)
        data = pd.DataFrame(data)
        data = data.sample(frac=1).reset_index(drop=True)

        labels = data.category.tolist()
        label_encoder = preprocessing.LabelEncoder()
        encoded = label_encoder.fit_transform(np.array(labels))
        data.category = encoded

        trainDf, valDf = train_test_split(data, test_size=0.3, shuffle=True)
        valDf, testDf = train_test_split(valDf, test_size=0.2, shuffle=True)

        temp = np.array([idx for idx in range(len(trainDf))])
        trainDf['index'] = temp
        trainDf.set_index('index', inplace=True)

        temp = np.array([idx for idx in range(len(valDf))])
        valDf['index'] = temp
        valDf.set_index('index', inplace=True)

        temp = np.array([idx for idx in range(len(testDf))])
        testDf['index'] = temp
        testDf.set_index('index', inplace=True)

        trainDf.to_csv(os.path.join(self.save_path, f'train_{self.worker}.csv'), index=None)
        valDf.to_csv(os.path.join(self.save_path, f'val_{self.worker}.csv'), index=None)
        testDf.to_csv(os.path.join(self.save_path, f'test_{self.worker}.csv'), index=None)
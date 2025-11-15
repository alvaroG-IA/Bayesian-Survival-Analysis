import numpy as np
from data.dataset import HeartHealthDataset
from models.logistic_bayes import LogisticBayesModel
from sklearn.metrics import accuracy_score

def main():

    data_path = 'data/fallo_cardiaco.csv'
    
    dataset = HeartHealthDataset(data_path)
    X, y = dataset.preproces(mode='std')

    model = LogisticBayesModel()
    w = model.fit(X, y)

    pred =  np.where(np.dot(X, w) >= 0.5, 1, 0)
    
    acc = accuracy_score(pred, y)
    print(f'Accuracy {acc}')

if __name__ == '__main__':
    main()
    

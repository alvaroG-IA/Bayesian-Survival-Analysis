from src.data.dataset import HeartHealthDataset
from src.models.logistic_bayes import LogisticBayesModel

def main():
    data_path = 'data/fallo_cardiaco.csv'
    dataset = HeartHealthDataset(data_path)

    model = LogisticBayesModel()
    
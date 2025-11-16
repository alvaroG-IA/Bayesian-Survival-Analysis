from src.data.dataset import HeartHealthDataset
from src.models.logistic_bayes import LogisticBayesModel
from src.utils.helpers import plot_post_distribuitions
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():

    data_path = 'data/fallo_cardiaco.csv'
    dataset = HeartHealthDataset(data_path)
    col_names = dataset.get_col_names()

    X, y = dataset.get_raw_data()

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

    scaler = StandardScaler()
    Xtrain_std = scaler.fit_transform(Xtrain)
    Xtest_std = scaler.transform(Xtest)

    model = LogisticBayesModel()

    model.fit(Xtrain_std, ytrain, iterations=50000, burn_in=1000)

    pred = model.predict(Xtest_std)

    acc = accuracy_score(pred, ytest)
    print(f'\n[Accuracy sobre el conjunto de Test] {acc:.4f}')
    print(f'La tasa de aceptación de muestras durante el entrenamiento ha sido de: {model.acceptance_ratio:.5}%')

    samples = model.samples_
    w_mean = model.w_mean_
    w_std = model.w_std_
    w_ci = model.w_ci_

    plot_post_distribuitions(samples, w_mean, w_std, X.shape[1], col_names)

    for i in range(X.shape[1]):
        print(f'\n -- [Valores para la variable "{col_names[i]}"] -- ')
        print(f'* Valor medio/propuesto: {w_mean[i]:.4}')
        print(f'* Desviacion estándar: {w_std[i]:.4}')
        print(f'*CI (95% de conianza): ({w_ci[0][i]:.4}, {w_ci[1][i]:.4})')


if __name__ == '__main__':
    main()
    

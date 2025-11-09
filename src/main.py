import pandas as pd
from sklearn.model_selection import train_test_split

data_path = 'data/fallo_cardiaco.csv'
df = pd.read_csv(data_path)

labels = df['DEATH_EVENT']
data = df.drop(['DEATH_EVENT'], axis=1)

Xtrain, Xtest, ytrain, ytest = train_test_split(data, labels, test_size=0.15, random_state=42, stratify=labels)
import pickle
from sklearn.pipeline import Pipeline


def save_pipeline(pipeline: Pipeline, path: str):
    try:
        with open(path, 'wb') as f:
            pickle.dump(pipeline, f)
        print(f'\n✅ Pipeline guardado correctamente en {path}')
    except Exception as e:
        print(f'Error al guardar con pickle {e}')


def load_pipeline(path: str):
    try:
        with open(path, 'rb') as f:
            print('✅ Pipeline cargado correctamente')
            return pickle.load(f)
    except Exception as e:
        print(f'Error al cargan con pickle {e}')
        return None

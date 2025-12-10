import pickle
from sklearn.pipeline import Pipeline


def save_pipeline(pipeline: Pipeline, path: str) -> None:
    """
    Function to save a pipeline as a pickle object.
    """
    try:
        with open(path, 'wb') as f:
            pickle.dump(pipeline, f)
        print(f'\n✅ Pipeline successfully saved to {path}')
    except Exception as e:
        print(f'Error saving with pickle: {e}')


def load_pipeline(path: str) -> Pipeline or None:
    """
    Function to load a pipeline from a pickle object.
    """
    try:
        with open(path, 'rb') as f:
            print('✅ Pipeline successfully loaded')
            return pickle.load(f)
    except Exception as e:
        print(f'Error loading with pickle: {e}')
        return None

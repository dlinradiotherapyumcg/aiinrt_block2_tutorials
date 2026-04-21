import numpy as np

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def get_digits_data(SEED, max_samples=500):    

    digits = load_digits()

    X_all = digits.images
    y_all = digits.target


    # # Randomly pick 100 samples of each class for test set
    # indices_test = []
    # for class_idx in range(len(set(y_all))):
    #     class_indices = np.where(y_all == class_idx)[0]
    #     indices_test.extend(np.random.choice(class_indices, size=100, replace=False))

    # indices_test = np.array(indices_test)
    # indices_train = np.setdiff1d(np.arange(len(y_all)), indices_test)

    # if max_samples is not None:
    #     indices_train = indices_train[:max_samples]

    # X_train = X_all[indices_train].reshape((len(indices_train), -1))
    # y_train = y_all[indices_train]
    # X_test = X_all[indices_test].reshape((len(indices_test), -1))
    # y_test = y_all[indices_test]


    X_train, X_test, y_train, y_test = train_test_split_dataset(X_all, y_all, max_samples=max_samples, SEED=SEED)



    # # Limit the number of samples
    # if max_samples is not None:
    #     X_all = X_all[:max_samples]
    #     y_all = y_all[:max_samples]

    # n_samples = len(y_all)
    # X_all_flattened = X_all.reshape((n_samples, -1))

    # X_train_all, X_test, y_train_all, y_test = train_test_split(
    #     X_all_flattened, y_all, test_size=0.5, random_state=SEED, stratify=y_all
    # )

    return X_train, X_test, y_train, y_test





def train_test_split_dataset(X_all, y_all, max_samples=500, SEED=42):

    # Randomly pick 100 samples of each class for test set
    indices_test = []
    for class_idx in range(len(set(y_all))):
        class_indices = np.where(y_all == class_idx)[0]
        indices_test.extend(np.random.choice(class_indices, size=100, replace=False))

    indices_test = np.array(indices_test)
    indices_train = np.setdiff1d(np.arange(len(y_all)), indices_test)

    if max_samples is not None:
        indices_train = indices_train[:max_samples]

    X_train = X_all[indices_train].reshape((len(indices_train), -1))
    y_train = y_all[indices_train]
    X_test = X_all[indices_test].reshape((len(indices_test), -1))
    y_test = y_all[indices_test]

    return X_train, X_test, y_train, y_test
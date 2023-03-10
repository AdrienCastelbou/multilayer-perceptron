import numpy as np
from constants import features_to_exclude
import pickle


def normalize(x):
    norm_x = np.array([])
    for col in x.T:
        mean_col = np.mean(col)
        std_col = np.std(col)
        n_col = ((col - mean_col) / std_col).reshape((-1, 1))
        if norm_x.shape == (0,):
            norm_x = n_col
        else:
            norm_x = np.hstack((norm_x, n_col))
    return norm_x


def binarise(X):
    bin_x = np.zeros(X.shape).astype(int)
    for i in range(len(X)):
        if X[i] == "M":
            bin_x[i] = 1
        elif X[i] == "B":
            bin_x[i] = 0
    return bin_x

def data_spliter(x, y, proportion):
    try:
        if type(x) != np.ndarray or type(y) != np.ndarray or type(proportion) != float:
            return None
        n_train = int(proportion * x.shape[0])
        n_test = int((1-proportion) * x.shape[0]) + 1
        perm = np.random.permutation(len(x))
        s_x = x[perm]
        s_y = y[perm]
        x_train, y_train = s_x[:n_train], s_y[:n_train]
        x_test, y_test =  s_x[-n_test:], s_y[-n_test:]
        return x_train, x_test, y_train, y_test
    except:
        return None
    

def data_preprocessing(df):
    df.drop(df.columns[0], axis=1, inplace=True)
    col_names = ["State"]
    for i in range(1, 31):
        col_names.append(f"f{i}")
    df.columns = col_names
    X = df[[feature for feature in df.columns if feature not in features_to_exclude]].to_numpy()
    y = binarise(df["State"].to_numpy())
    y = y.reshape((y.shape[0], -1))
    return X, y
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from utils import *


def load_args():
    if len(sys.argv) != 3:
        raise Exception("Error : Wrong number of arguments -> Usage : python3 describe.py path/to/dataset.csv")
    f = open(sys.argv[1], 'rb')
    model = pickle.load(f)
    df = pd.read_csv(sys.argv[2])
    return model, df

def plot_model_performance(y, y_hat):
    plt.scatter(range(0, len(y)), y, label='values')
    plt.scatter(range(0, len(y)), y_hat, [10], label='preds',)
    plt.legend(loc='best')
    plt.grid()
    plt.show()

def main():
    model, df = load_args()
    X, y = data_preprocessing(df)
    X = normalize(X)
    preds = model.predict(X)
    print(f"loss: {model.loss(preds, y)}, accuracy: {np.mean(preds == y)}")
    plot_model_performance(y, preds)
if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from MLP import MLP
from utils import *
import pickle
from constants import features_to_exclude

def load_dataset():
    if len(sys.argv) != 2:
        raise Exception("Error : Wrong number of arguments -> Usage : python3 describe.py path/to/dataset.csv")
    df = pd.read_csv(sys.argv[1])
    return df

def save_models(results):
    file = open('model.pickle', 'wb')
    pickle.dump(results, file)
    file.close()

def hist(df):
    M_df = df[df["State"] == "M"]
    B_df = df[df["State"] == "B"]
    for feature in df.columns:
        if feature != "State":
            plt.title(feature)
            plt.hist(M_df[feature].to_numpy(), bins=20, label="M", alpha=0.5)
            plt.hist(B_df[feature].to_numpy(), bins=20, label="B", alpha=0.5)
            plt.legend()
            plt.show()

def show_pair_plot(df):
    df = df[[feature for feature in df.columns if feature not in features_to_exclude]]
    print(df.columns)
    sns.pairplot(df,  diag_kind="hist", hue="State", markers='.', height=2)
    plt.show()

    
def train_model(X, y):
    myMLP = MLP(max_iter=1000)
    loss_log, val_log = myMLP.fit(normalize(X), y)
    save_models(myMLP)
    plt.plot(loss_log,label='loss')
    plt.plot(val_log,label='val_loss')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

def main():
    df = load_dataset()
    X, y = data_preprocessing(df)
    train_model(X, y)


if __name__ == "__main__":
    main()
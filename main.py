import pandas as pd
import numpy as np
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from MLP import MLP


features_to_exclude = ["State", "f3", "f5", "f9", "f10", "f12", "f15", "f16", "f17", "f18", "f19", "f20", "f22", "f25", "f30"]
def load_dataset():
    if len(sys.argv) != 2:
        raise Exception("Error : Wrong number of arguments -> Usage : python3 describe.py path/to/dataset.csv")
    df = pd.read_csv(sys.argv[1])
    return df

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

    
def train_model(df):
    X = df[[feature for feature in df.columns if feature not in features_to_exclude]].to_numpy()
    y = binarise(df["State"].to_numpy())
    y = y.reshape((y.shape[0], -1))
    myMLP = MLP()
    myMLP.fit(normalize(X[np.arange(19,24)]), y[np.arange(19, 24)])
    

def main():
    df = load_dataset()
    df.drop(df.columns[0], axis=1, inplace=True)
    col_names = ["State"]
    for i in range(1, 31):
        col_names.append(f"f{i}")
    df.columns = col_names
    train_model(df)


if __name__ == "__main__":
    main()
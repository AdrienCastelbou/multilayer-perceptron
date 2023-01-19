import pandas as pd
import numpy as np
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

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

def binarise(X):
    bin_x = np.zeros(X.shape)
    for i in range(len(X)):
        if X[i] == "M":
            bin_x[i] = 1
        elif X[i] == "B":
            bin_x[i] = 0
    return bin_x

def train(df):
    X = df[[feature for feature in df.columns if feature not in features_to_exclude]].to_numpy()
    y = binarise(df["State"].to_numpy())
    y = df["State"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)
    clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
    clf.fit(X_train, y_train)
    scores = cross_val_score(clf, X, y, cv=5)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    print(clf.score(X_test, y_test))
    

def main():
    df = load_dataset()
    df.drop(df.columns[0], axis=1, inplace=True)
    col_names = ["State"]
    for i in range(1, 31):
        col_names.append(f"f{i}")
    df.columns = col_names
    train(df)

if __name__ == "__main__":
    main()
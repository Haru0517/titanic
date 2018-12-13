import pandas as pd
import numpy as np
from sklearn import tree


train = pd.read_csv("../dataset/train.csv")
test = pd.read_csv("../dataset/test.csv")


def kesson_table(df):
    null_val = df.isnull().sum()
    percent = 100 * df.isnull().sum()/len(df)
    kesson_table = pd.concat([null_val, percent], axis=1)
    kesson_table_ren_columns = kesson_table.rename(
        columns={0: '欠損数', 1: '%'})
    return kesson_table_ren_columns


def preprocessing():
    # 欠損値を埋める
    train["Age"] = train["Age"].fillna(train["Age"].median())
    train["Embarked"] = train["Embarked"].fillna("S")
    # print(kesson_table(train))

    # 文字列を数字に変換
    train["Sex"][train["Sex"] == "male"] = 0
    train["Sex"][train["Sex"] == "female"] = 1
    train["Embarked"][train["Embarked"] == "S"] = 0
    train["Embarked"][train["Embarked"] == "C"] = 1
    train["Embarked"][train["Embarked"] == "Q"] = 2
    # print(train.head(10))

    test["Age"] = test["Age"].fillna(test["Age"].median())
    test["Sex"][test["Sex"] == "male"] = 0
    test["Sex"][test["Sex"] == "female"] = 1
    test["Embarked"][test["Embarked"] == "S"] = 0
    test["Embarked"][test["Embarked"] == "C"] = 1
    test["Embarked"][test["Embarked"] == "Q"] = 2
    test.Fare[152] = test.Fare.median()
    # print(test.head(10))

    return train, test


def tree_model():
    # 「train」の目的変数と説明変数の値を取得
    target = train["Survived"].values
    features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

    # 決定木の作成
    my_tree_one = tree.DecisionTreeClassifier()
    my_tree_one = my_tree_one.fit(features_one, target)

    # 「test」の説明変数の値を取得
    test_features = test[["Pclass", "Sex", "Age", "Fare"]].values

    # 「test」の説明変数を使って「my_tree_one」のモデルで予測
    my_prediction = my_tree_one.predict(test_features)

    # PassengerIdを取得
    PassengerId = np.array(test["PassengerId"]).astype(int)

    # my_prediction(予測データ）とPassengerIdをデータフレームへ落とし込む
    my_solution = pd.DataFrame(my_prediction, PassengerId, columns=["Survived"])

    # my_tree_one.csvとして書き出し
    my_solution.to_csv("../result/my_tree_one.csv", index_label=["PassengerId"])


def tree_model2():
    # 「train」の目的変数と説明変数の値を取得
    target = train["Survived"].values
    features_two = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

    # 決定木の作成とアーギュメントの設定
    max_depth = 10
    min_samples_split = 5
    my_tree_two = tree.DecisionTreeClassifier(max_depth=max_depth,
                                              min_samples_split=min_samples_split, random_state=1)
    my_tree_two = my_tree_two.fit(features_two, target)

    # tsetから「その2」で使う項目の値を取り出す
    test_features_2 = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

    # 「その2」の決定木を使って予測をしてCSVへ書き出す
    my_prediction_tree_two = my_tree_two.predict(test_features_2)
    PassengerId = np.array(test["PassengerId"]).astype(int)
    my_solution_tree_two = pd.DataFrame(my_prediction_tree_two, PassengerId, columns=["Survived"])
    my_solution_tree_two.to_csv("../result/my_tree_two.csv", index_label=["PassengerId"])


def main():
    preprocessing()
    tree_model2()


if __name__ == '__main__':
    main()

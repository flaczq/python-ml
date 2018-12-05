from sklearn import tree
from sklearn.datasets import load_iris
import numpy as np

print('----- START -----\n')


def machineLearingGoogle():
    # 0 - bumpy / apple
    # 1 - smooth / orange
    features = [[140, 1], [130, 1], [150, 0], [170, 0]]
    labels = [0, 0, 1, 1]
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(features, labels)
    print('apple' if clf.predict([[170, 0]]) == 0 else 'orange')


def mlIris():
    # 0 - setosa
    # 1 - versicolor
    # 2 - virginica
    iris = load_iris()
    test_idx = [0, 50, 100]

    train_target = np.delete(iris.target, test_idx)
    train_data = np.delete(iris.data, test_idx, axis=0)

    test_target = iris.target[test_idx]
    test_data = iris.data[test_idx]

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_data, train_target)

    print(test_target)
    print(clf.predict(test_data))


# machineLearingGoogle()
mlIris()

print('\n----- STOOP -----')

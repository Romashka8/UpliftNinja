# ----------------------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from typing import List

# ----------------------------------------------------------------------------------------------------------------------------------------------------------


class Node:
    """
    Класс узла решающешо дерева.
    """

    def __init__(self,
                 feature=None,
                 threshold=None,
                 left=None,
                 right=None,
                 gain=None,
                 value=None
                 ):
        """
        Инициализирует новый экземпляр класса Node.

        Аргументы:
            feature: признак, используемый для разделения в этом узле. Значение по умолчанию равно None.
            threshold: порог, используемый для разделения на этом узле. Значение по умолчанию равно None.
            left: левый дочерний узел. По умолчанию равно None.
            right: правый дочерний узел. По умолчанию равно None.
            gain: прирост информации при разделении. По умолчанию равно None.
            value: если этот узел является конечным, этот атрибут представляет прогнозируемое значение
                для целевой переменной. По умолчанию установлено значение "Нет".
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gain = gain
        self.value = value


class DecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 min_samples: int = 2,
                 max_depth: int = 2
                 ):
        """
        Конструктор для класса DecisionTree.

        Параметры:
            min_samples (int): минимальное количество выборок, необходимое для разделения внутреннего узла.
            max_depth (int): максимальная глубина дерева решений.
        """
        self.min_samples = min_samples
        self.max_depth = max_depth

    def split_data(self,
                   dataset: np.ndarray,
                   feature: int,
                   threshold: float
                   ) -> List[np.ndarray]:
        """
        Разбивает данный набор данных на два набора данных на основе заданного признака и порогового значения.

        Параметры:
            dataset (ndarray): входной набор данных.
            feature (int): индекс признака, по которому будет произведено разделение.
            порог (float): пороговое значение для разделения объекта.

        Возвращается:
            left_dataset (ndarray): подмножество набора данных со значениями, меньшими или равными пороговому значению.
            right_dataset (ndarray): подмножество набора данных со значениями, превышающими пороговое значение.
        """

        left_dataset = []
        right_dataset = []

        for row in dataset:
            if row[feature] <= threshold:
                left_dataset.append(row)
            else:
                right_dataset.append(row)

        left_dataset = np.array(left_dataset)
        right_dataset = np.array(right_dataset)

        return left_dataset, right_dataset

    def entropy(self,
                y: np.ndarray
                ) -> float:
        """
        Вычисляет энтропию заданных значений меток.

        Параметры:
            y (ndarray): Ввод значений меток.

        Возвращается:
            энтропия (с плавающей точкой): энтропия заданных значений меток.
        """
        entropy = 0

        labels = np.unique(y)
        for label in labels:
            label_example = y[y == label]
            pl = len(label_example) / len(y)
            entropy += -pl * np.log2(pl)

        return entropy

    def information_gane(self,
                         parent: np.array,
                         left: np.array,
                         right: np.array) -> float:
        """
        Вычисляет информацию, полученную в результате разделения родительского набора данных на два набора данных.

        Параметры:
            parent (ndarray): ввод родительского набора данных.
            left (ndarray): подмножество родительского набора данных после разделения на объект.
            right (ndarray): подмножество родительского набора данных после разделения на объект.

        Возвращается:
            information_gain (с плавающей точкой): прирост информации при разделении.
        """
        information_gane = 0
        parent_entropy = self.entropy(parent)
        weight_left = len(left) / len(parent)
        weight_right = len(right) / len(parent)
        entropy_left, entropy_right = self.entropy(left), self.entropy(right)
        weighted_entropy = weight_left * entropy_left + weight_right * entropy_right
        information_gane = parent_entropy - weighted_entropy
        return information_gane

    def best_split(self,
                   dataset: np.ndarray,
                   num_samples: int,
                   num_features: int
                   ) -> dict:
        """
        Находит наилучшее разделение для данного датасета.

        Аргументы:
        dataset (ndarray): Набор данных для разделения.
        num_samples (int): Количество выборок в наборе данных.
        num_features (int): количество объектов в наборе данных.

        Возвращается:
        dict: Словарь с наилучшим разделением индекса признаков, порогового значения, коэффициента усиления,
              левого и правого наборов данных.
        """
        best_split = {
            "gain": -1,
            "feature": None,
            "threshold": None
        }

        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            thresholds = np.unique(feature_values)
            for threshold in thresholds:
                left_dataset, right_dataset = self.split_data(
                    dataset, feature_index, threshold)
                if len(left_dataset) and len(right_dataset):
                    y, left_y, right_y = dataset[:, -
                                                 1], left_dataset[:, -
                                                                  1], right_dataset[:, -
                                                                                    1]
                    information_gain = self.information_gane(
                        y, left_y, right_y)
                    if information_gain > best_split["gain"]:
                        best_split["feature"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["left_dataset"] = left_dataset
                        best_split["right_dataset"] = right_dataset
                        best_split["gain"] = information_gain

        return best_split

    def calculate_leaf_value(self,
                             y: list | np.array) -> int:
        """
        Вычисляет наиболее часто встречающееся значение в заданном списке значений y.

        Аргументы:
            y (список): список значений y.

        Возвращается:
            Наиболее часто встречающееся значение в списке.
        """
        y = list(y)
        most_occuring_value = max(y, key=y.count)
        return most_occuring_value

    def build_tree(self,
                   dataset: np.ndarray,
                   current_depth: int=0
                   ) -> Node:
        """
        Рекурсивно строит дерево решений из заданного набора данных.

        Аргументы:
        dataset (ndarray): Набор данных, из которого будет построено дерево.
        current_depth (int): Текущая глубина дерева.

        Возвращается:
        Node: корневой узел построенного дерева решений.
        """
        X, y = dataset[:, :-1], dataset[:, -1]
        n_samples, n_features = X.shape

        if n_samples >= self.min_samples and current_depth <= self.max_depth:
            best_split = self.best_split(dataset, n_samples, n_features)
            if best_split["gain"]:
                left_node = self.build_tree(
                    best_split["left_dataset"], current_depth + 1)
                right_node = self.build_tree(
                    best_split["right_dataset"], current_depth + 1)
                return Node(best_split["feature"], best_split["threshold"],
                            left_node, right_node, best_split["gain"])

        leaf_value = self.calculate_leaf_value(y)
        return Node(value=leaf_value)

    def fit(self,
            X: np.ndarray,
            y: np.ndarray) -> DecisionTreeClassifier:
        """
        Строит и подгоняет дерево решений к заданным значениям X и y.

        Аргументы:
        X (ndarray): матрица признаков.
        y (ndarray): целевые значения.
        """
        dataset = np.concatenate((X, y), axis=1)
        self.root = self.build_tree(dataset)
        return self

    def predict(self,
                X: np.ndarray) -> np.array:
        """
        Предсказывает метки классов для каждого экземпляра в матрице объектов X.

        Аргументы:
        X (ndarray): Матрица объектов, для которой нужно делать прогнозы.

        Возвращается:
            Список предсказанных меток классов.
        """
        predictions = []
        for x in X:
            prediction = self.make_prediction(x, self.root)
            predictions.append(prediction)
        predictions = np.array(predictions)
        return predictions

    def make_prediction(self,
                        x: np.array,
                        node: Node) -> float:
        """
        Обходит дерево решений, чтобы предсказать целевое значение для данного вектора признаков.

        Аргументы:
            x (ndarray): вектор признаков, для которого нужно предсказать целевое значение.
            node: вычисляемый текущий узел.

        Возвращается:
            Прогнозируемое целевое значение для данного вектора признаков.
        """
        if node.value is not None:
            return node.value
        else:
            feature = x[node.feature]
            if feature <= node.threshold:
                return self.make_prediction(x, node.left)
            else:
                return self.make_prediction(x, node.right)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # искусственные данные
    X = np.array([[1, 2],
                [2, 3],
                [3, 1],
                [4, 5],
                [5, 4],
                [6, 6]])
    y = np.array([0, 0, 0, 1, 1, 1])
    y = y.reshape((X.shape[0], 1))

    clf = DecisionTreeClassifier(min_samples=2, max_depth=3)
    clf.fit(X, y)

    predictions = clf.predict(X)
    print(predictions)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------

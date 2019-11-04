import numpy as np
from os import listdir
from os.path import isfile, join
from PIL import Image
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Perceptron:

    def __init__(self):

        self.weights = None

    def fit(self, X, y, lr=0.01, epochs=50):

        weights = np.random.rand(X.shape[1], )
        weig = []

        for epoch in range(epochs):

            predicted = []
            for i_index, sample in enumerate(X):
                y_hat = self.make_step(sample, weights)
                predicted.append(y_hat)
                # error = y_hat - y
                # grad = y_hat * (1 - y_hat)
                # weig = error * grad

                for j_index, feature in enumerate(weights):
                    delta = lr * (y[i_index] - y_hat)
                    # weights = self._calculate_error1(y, predicted)
                    delta = delta * sample[j_index - 1]
                    weights[j_index - 1] = weights[j_index - 1] + delta

            print('[Epoch {ep}] Accuracy of train set: {acc}'.format(ep=epoch, acc=self._calculate_accuracy(y, predicted)))

        self.weights = weights

    def _calculate_accuracy1(self, actual, predicted):
        error = np.array(actual) - np.array(predicted)
        grad = np.array(actual) * (1 + np.array(actual))
        weights = np.array(error) * np.array(grad)
        return weights

    def make_step(self, x, w):

        res = self._sum(x, w)
        return 1 if res > 0.0 else 0.0

    def _calculate_accuracy(self, actual, predicted):

        return sum(np.array(predicted) == np.array(actual)) / float(len(actual))

    def predict(self, x):

        res = self._sum(x, self.weights)
        return 1 if res > 0.0 else 0.0

    def group_preds(self, x):

        all_predictions = []
        for row in range(x.shape[0]):
            all_predictions.append(self.predict(x[row]))
        return all_predictions

    def _sum(self, x, w):

        return np.sum(np.dot(x, np.transpose(w)))

    def save_weights(self, filename):

        with open(filename, 'wb') as file:
            pickle.dump(self.weights, file)

    def load_weights(self, filename):

        with open(filename, 'rb') as file:
            self.weights = pickle.load(file)


if __name__ == '__main__':

    p = Perceptron()
    norma_path = 'D:/практика/norma'
    adenoma_path = 'D:/практика/adenoma'
    norma_TEST_path = 'D:/практика/normaTest'
    adenoma_TEST_path = 'D:/практика/adenomaTest'
    n_samples = 100
    n_TEST_samples = 5

    norma_path_files = [file for file in listdir(norma_path) if isfile(join(norma_path, file))][:n_samples]
    adenoma_path_files = [file for file in listdir(adenoma_path) if isfile(join(adenoma_path, file))][:n_samples]

    norma_TEST_path_files = [file for file in listdir(norma_TEST_path) if isfile(join(norma_TEST_path, file))][
                            :n_TEST_samples]
    adenoma_TEST_path_files = [file for file in listdir(adenoma_TEST_path) if isfile(join(adenoma_TEST_path, file))][
                              :n_TEST_samples]

    images, target = [], []  # массивы исходных и нужных(получаемых) изображений
    image_size = (600, 600)  # размер к которому приводим изображение
    thresh = 125  # значение порога изображения
    test_size = 0

    for norma_file, adenoma_file in zip(norma_path_files, adenoma_path_files):
        image = Image.open(join(norma_path, norma_file)).convert('L')
        image = np.asarray(image.resize(image_size)).flatten()  # переменные 1\0 возвращает сглаженный одномерный массив

        image[image > thresh] = 1  # нужная вещь для выделения блеклых элементов(яркость) через пороговое значение?????
        image[image > thresh] <= 0  # из инета ????

        images.append(image)  # добавляет элемент image в конец списка images
        target.append(1)  # в массив заносится 1(норма)
        image = Image.open(join(adenoma_path, adenoma_file)).convert('L')  # переменные 1\0
        image = np.asarray(image.resize(image_size)).flatten()  # преобразование изображения в нужный размер

        image[image > thresh] = 1  # из инета ????
        image[image > thresh] <= 0  # из инета ????

        images.append(image)  # добавляет элемент image в конец списка images
        target.append(0)  # в массив заносится 0(аденома)

    X = np.vstack(images)  # вертикальное преобразование в один массив
    X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.1, random_state=42)
    print(target)
    print(y_train)
    p.fit(X_train, y_train, lr=0.001, epochs=5)
    preds = p.group_preds(X_test)
    print(y_test)
    print(preds)
    # for images in enumerate(X_test):
    #   if preds == 1: print('this is norma - {0}')
    #   else: print('this is adenorma - {0}')

    acc = accuracy_score(y_test, preds)
    print('Accuracy on val(test) set - {0}'.format(acc))

    # img = Image.open(r'D:../norma/65.png')
    # pr = p.group_preds(img)
    # acc = accuracy_score(y_test, pr)
    # print('Accuracy on 1 image - {0}'.format(acc))

    p.save_weights('D:/практика/weights.pickle')






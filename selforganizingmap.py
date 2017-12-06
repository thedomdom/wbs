import numpy as np
import sklearn.datasets
import sklearn.metrics as metrics
import random
import sys


class SOM:
    def __init__(self, n_rows=5, n_cols=5, iterations_max=10, alpha_0=0.5, sigma_0=1, neighbor_function='gauss'):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.t_max = iterations_max
        self.alpha_0 = alpha_0
        self.sigma_0 = sigma_0
        self.neighbor_function = neighbor_function
        self.neurons = None

        # Erstelle Map für die Klassen zu den Neuronen
        self.neuron_classes = {}
        for x_coord in range(self.n_rows):
            self.neuron_classes[x_coord] = {}
            for y_coord in range(self.n_cols):
                self.neuron_classes[x_coord][y_coord] = {}

        self.progress_bar = list(range(11))

    def fit(self, X, y):
        # 1. Initialisierung der Neuronen (zufällig, minmax)
        self.neurons = np.random.uniform(size=(self.n_rows, self.n_cols, len(X[0])))

        for t in range(self.t_max):
            self.print_progress(t)

            # Ein Datenpunkt wird zufällig dem Datensatz entnommen
            x_index = random.choice(range(len(X)))
            x = X[x_index]

            # 2. Competition: Das nächste Neuron, das zum aktuellen Datenpunkt passt (BMU)
            bmu_coord = self.best_matching_unit(x)

            # 3. Cooperation: Benachbarte Neuronen werden über einen "Nachbarschafts-Radius" gefunden
            neighbor_coords = self.get_neighbor_coordinates(bmu_coord, t)

            # 4. Adaptation: Benachbarte Neuronen werden abhängig von ihrem Abstand
            # auf der Karte zum Datenpunkt "hingezogen" --> Anpassung der Gewichte
            for neighbor_coord in neighbor_coords:
                # print(self.neurons[neighbor_coord[0]][neighbor_coord[1]])
                neighbor = self.neurons[neighbor_coord[0]][neighbor_coord[1]]
                neighbor = neighbor + self.alpha(t) * self.h(neighbor_coord, bmu_coord, t) * (x - neighbor)
                self.neurons[neighbor_coord[0]][neighbor_coord[1]] = neighbor

        # Den Neuronen die jeweiligen Klassen zuordnen
        for x_index in range(len(X)):
            x = X[x_index]
            x_class = y[x_index]
            x_coord = self.classify(x)
            # Der BMU die Klasse des Datenpunkts zuweisen
            self.assign_class(x_coord, x_class)

        # Finde zu jedem Neuron die meist-zugewiesene Klasse
        for x_coord in range(self.n_rows):
            for y_coord in range(self.n_cols):
                self.set_best_class(x_coord, y_coord)

    def set_best_class(self, x_coord, y_coord):
        classes_dict = self.neuron_classes[x_coord][y_coord]
        sorted_amounts = sorted(classes_dict.values(), reverse=True)

        self.neuron_classes[x_coord][y_coord] = -1

        for class_number, amount in classes_dict.items():
            if amount == sorted_amounts[0]:
                self.neuron_classes[x_coord][y_coord] = class_number
                break

    def assign_class(self, coord, coord_class):
        x_coord, y_coord = coord[0], coord[1]
        if coord_class in self.neuron_classes[x_coord][y_coord]:
            self.neuron_classes[x_coord][y_coord][coord_class] += 1
        else:
            self.neuron_classes[x_coord][y_coord][coord_class] = 1

    def alpha(self, t):
        return self.alpha_0 * (1 - t / self.t_max)

    def sigma(self, t):
        return self.sigma_0 * (1 - t / self.t_max)

    def h(self, c1, c2, t):
        return np.exp(- np.linalg.norm(c1 - c2) / (2 * self.sigma(t) ** 2))

    def best_matching_unit(self, x):
        coordinate = None
        dist_min = sys.float_info.max
        for x_coord, neuron_col in enumerate(self.neurons):
            for y_coord, neuron in enumerate(neuron_col):
                # print(neuron)
                dist = np.linalg.norm(neuron - x)
                if dist < dist_min:
                    dist_min = dist
                    coordinate = np.array([x_coord, y_coord])
                    # print(neuron)
                    # print(dist_min)
                    # print(self.neurons[coordinates[0]][coordinates[1]])
        return coordinate

    def best_matching_unit_performant(self, x):
        # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.copy.html
        # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.amin.html#numpy.amin
        pass

    def get_neighbor_coordinates(self, coordinate, t):
        neighbor_coordinates = []
        for x_coord, neuron_col in enumerate(self.neurons):
            for y_coord, neuron in enumerate(neuron_col):
                dist = np.linalg.norm(np.array([x_coord, y_coord]) - coordinate)
                # setting threshold depending on neighbor function
                if self.neighbor_function == 'gauss':
                    threshold = self.sigma(t)
                else:
                    threshold = self.n_rows * self.n_cols / int(self.neighbor_function)
                # appending neurons inside threshold as neighbors
                if dist <= threshold:
                    neighbor_coordinates.append([x_coord, y_coord])
        return neighbor_coordinates

    def classify(self, x):
        return self.best_matching_unit(x)

    def predict(self, X):
        y_predicted = []
        for x in X:
            x_coord, y_coord = self.classify(x)
            y_predicted.append(self.neuron_classes[x_coord][y_coord])
        return y_predicted

    def score(self, X, y_true):
        return metrics.accuracy_score(y_true, self.predict(X))

    def print_progress(self, t):
        if t/(self.t_max-1) >= self.progress_bar[0] / 10:
            print(self.progress_bar[0] * 10, "%", sep='', end=' ', flush=True)
            self.progress_bar.pop(0)

            if t == self.t_max-1:
                print()


if __name__ == "__main__":
    (X, y) = sklearn.datasets.load_iris(return_X_y=True)
    indices_train = np.random.randint(150, size=110)
    indices_evaluate = [index for index in list(range(150)) if (index not in list(indices_train))]

    print(X.shape)
    print(y.ravel().shape)

    my_som = SOM(n_rows=7, n_cols=7, iterations_max=5000, alpha_0=0.9, sigma_0=1, neighbor_function='8')
    my_som.fit(X[indices_train, :], y[indices_train])

    print(my_som.score(X[indices_evaluate], y[indices_evaluate]))

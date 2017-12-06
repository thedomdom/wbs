import numpy as np
import sklearn.datasets
import random
import sys


class SOM:
    def __init__(self, n_rows=5, n_cols=5, iterations_max=10, alpha_0=0.5, sigma_0=1):
        self.nRows = n_rows
        self.nCols = n_cols
        self.t_max = iterations_max
        self.alpha_0 = alpha_0
        self.sigma_0 = sigma_0
        self.neurons = None

        # Erstelle Map für die Klassen zu den Neuronen
        self.neuron_classes = {}
        for x in range(n_rows):
            self.neuron_classes[x] = {}
            for y in range(n_cols):
                self.neuron_classes[x][y] = {}

    def fit(self, X, y):
        # 1. Initialisierung der Neuronen (zufällig, minmax)
        # print(len(X[0]))
        self.neurons = np.random.uniform(size=(self.nRows, self.nCols, len(X[0])))

        for t in range(self.t_max):
            # Ein Datenpunkt wird zufällig dem Datensatz entnommen
            x_index = random.choice(range(len(X)))
            x = X[x_index]
            x_class = y[x_index]

            # 2. Competition: Das nächste Neuron, das zum aktuellen Datenpunkt passt (BMU)
            bmu_coord = self.best_matching_unit(x)

            # Der BMU die Klasse des Datenpunkts zuweisen
            self.assign_class(bmu_coord, x_class)

            # 3. Cooperation: Benachbarte Neuronen werden über einen "Nachbarschafts-Radius" gefunden
            neighbor_coords = self.get_neighbor_coordinates(bmu_coord)

            # 4. Adaptation: Benachbarte Neuronen werden abhängig von ihrem Abstand
            # auf der Karte zum Datenpunkt "hingezogen" --> Anpassung der Gewichte
            for neighbor_coord in neighbor_coords:
                # print(self.neurons[neighbor_coord[0]][neighbor_coord[1]])
                neighbor = self.neurons[neighbor_coord[0]][neighbor_coord[1]]
                neighbor = neighbor + self.alpha(t) * self.h(neighbor_coord, bmu_coord, t) * (x - neighbor)
                self.neurons[neighbor_coord[0]][neighbor_coord[1]] = neighbor

    def assign_class(self, coord, coord_class):
        x, y = coord[0], coord[1]
        if coord_class in self.neuron_classes[x][y]:
            self.neuron_classes[x][y][coord_class] += 1
        else:
            self.neuron_classes[x][y][coord_class] = 1

    def alpha(self, t):
        return self.alpha_0 * (1 - t / self.t_max)

    def h(self, c1, c2, t):
        sigma_t = self.sigma_0 * (1 - t / self.t_max)
        return np.exp(- np.linalg.norm(c1 - c2) / (2 * sigma_t ** 2))

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

    def get_neighbor_coordinates(self, coordinate):
        neighbor_coordinates = []
        for x_coord, neuron_col in enumerate(self.neurons):
            for y_coord, neuron in enumerate(neuron_col):
                dist = np.linalg.norm(np.array([x_coord, y_coord]) - coordinate)
                if dist < self.nRows * self.nCols / 8:
                    neighbor_coordinates.append([x_coord, y_coord])
        return neighbor_coordinates

    def classify(self, x):
        return self.best_matching_unit(x)


if __name__ == "__main__":
    (X, y) = sklearn.datasets.load_iris(return_X_y=True)

    print(X.shape)
    print(y.ravel().shape)

    my_som = SOM()
    my_som.fit(X, y)

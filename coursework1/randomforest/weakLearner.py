import numpy as np
from numpy.random import uniform, random_integers

__all__ = ["AxisAligned",
           "Linear",
           "Conic",
           "Parabola"]


class WeakLearner(object):
    def generate_all(self, points, count):
        return None

    def __str__(self):
        return None

    def run(self, point, test):
        return None


class AxisAligned(WeakLearner):
    """Axis aligned"""

    def __str__(self):
        return "AxisAligned"

    def generate_all(self, points, count):
        mins = [points.min(0)[i] for i in range(points.shape[1])]
        maxs = [points.max(0)[i] for i in range(points.shape[1])]
        tests = []
        uniform_list = [uniform(mins[i], maxs[i], count) for i in range(len(mins))]
        tests.extend(zip(*uniform_list,
                         random_integers(0, points.shape[1]-1, count)))  
        return np.array(tests)

    def run(self, point, test): # 변환 필요
        axis = test[point.shape[0]:]
        threshold = test[int(axis[0])]
        return point[int(axis[0])] > threshold

    def run_all(self, points, tests):
        def _run(test):
            axis = test[points.shape[1]:]
            threshold = test[int(axis[0])]
            return points[:, int(axis[0])] > threshold

        return np.array(list(map(_run, tests))).T
        
        


class Linear(WeakLearner):
    """Linear"""

    def __str__(self):
        return "Linear"

    def generate_all(self, points, count):
        mins = [points.min(0)[i] for i in range(points.shape[1])]
        maxs = [points.max(0)[i] for i in range(points.shape[1])]
        scale = abs(points.max() - points.min())
        tests = []
        uniform_list = [uniform(mins[i], maxs[i], count) for i in range(len(mins))]
        tests.extend(zip(*uniform_list,
                         uniform(-scale, scale, count),
                         uniform(-scale, scale, count),
                         uniform(-scale, scale, count),
                         random_integers(0, points.shape[1]-1, count),
                         random_integers(0, points.shape[1]-1, count)))
        return tests

    def run(self, point, test):
            A, B, C, axis1, axis2 = test[point.shape[0]:]
            x=(point[axis1] - test[axis1])
            y=(point[axis2] - test[axis2])
            return A*x + B * y + C > 0

    def run_all(self, points, tests):
        def _run(test):
            A, B, C, axis1, axis2 = test[points.shape[1]:]
            x=(points[:,axis1] - test[axis1])
            y=(points[:,axis2] - test[axis2])
            return A*x + B * y + C > 0

        return np.array(list(map(_run, tests))).T


class Conic(WeakLearner):
    """Non-linear: conic"""

    def __str__(self):
        return "Conic"

    def generate_all(self, points, count):
        mins = [points.min(0)[i] for i in range(points.shape[1])]
        maxs = [points.max(0)[i] for i in range(points.shape[1])]
        scale = max(points.max(), abs(points.min()))
        tests = []
        uniform_list = [uniform(mins[i], maxs[i], count) for i in range(len(mins))]
        tests.extend(zip(*uniform_list,
                         uniform(-scale, scale,count) * random_integers(0, 1, count),
                         uniform(-scale, scale,count) * random_integers(0, 1, count),
                         uniform(-scale, scale,count) * random_integers(0, 1, count),
                         uniform(-scale, scale,count) * random_integers(0, 1, count),
                         uniform(-scale, scale,count) * random_integers(0, 1, count),
                         uniform(-scale, scale,count) * random_integers(0, 1, count),
                         random_integers(0, points.shape[1]-1, count),
                         random_integers(0, points.shape[1]-1, count)))

        return tests

    def run(self, point, test):
        A, B, C, D, E, F, fidx1, fidx2 = test[point.shape[0]:]
        x = (point[fidx1] - test[fidx1])
        y = (point[fidx2] - test[fidx2])
        return (A * x * x + B * y * y + C * x * x + D * x + E * y + F) > 0
          
    def run_all(self, points, tests):
        def _run(test):
            A, B, C, D, E, F, fidx1, fidx2 = test[points.shape[1]:]
            x = (points[:,fidx1] - test[fidx1])
            y = (points[:,fidx2] - test[fidx2])
            return (A * x * x + B * y * y + C * x * x + D * x + E * y + F) > 0

        return np.array(list(map(_run, tests))).T


class Parabola(WeakLearner):
    """Non-linear: parabola"""

    def __str__(self):
        return "Parabola"

    def generate_all(self, points, count):
        mins = [points.min(0)[i] for i in range(points.shape[1])]
        maxs = [points.max(0)[i] for i in range(points.shape[1])]
        scale = abs(points.max() - points.min())
        tests = []
        uniform_list = [uniform(2 * mins[i], 2 * maxs[i], count) for i in range(len(mins))]
        tests.extend(zip(*uniform_list,
                         uniform(-scale, scale, count),
                         random_integers(0, points.shape[1]-1, count),
                         random_integers(0, points.shape[1]-1, count)))
        return tests

    def run(self, point, test):
            p, axis1, axis2 = test[point.shape[0]:]
            x=(point[axis1] - test[axis1])
            y=(point[axis2] - test[axis2])
            return x * x < p * y

    def run_all(self, points, tests):
        def _run(test):
            p, axis1, axis2 = test[points.shape[1]:]
            x=(points[:,axis1] - test[axis1])
            y=(points[:,axis2] - test[axis2])
            return x * x < p * y
        return np.array(list(map(_run, tests))).T


class FeatureExtractor(object):
    def __init__(self, learner, n_features):
        self.learner = learner
        self.n_features = n_features
        self.tests = []

    def fit_transform(self, points):
        self.tests = self.learner.generate_all(points, self.n_features)
        return self.apply_all(points)

    def apply(self, point):
        return np.array(list(map(lambda t: self.learner.run(point, t),self.tests)))

    def apply_all(self, points):
        return self.learner.run_all(points, self.tests)

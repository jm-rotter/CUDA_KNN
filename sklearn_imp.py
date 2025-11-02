import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import time
import sys

def read_csv_float(filename):
    return np.loadtxt(filename, delimiter=',', dtype=np.float32)

def read_csv_int(filename):
    return np.loadtxt(filename, delimiter=',', dtype=np.int32)

DISTANCE_MAP = {
    "L1": "manhattan",
    "L2": "euclidean",
    "LINF": "chebyshev",
    "COSINE": "cosine"
}

def create_knn(distance_type: str, k_neighbors: int) -> KNeighborsClassifier:
    distance_type = distance_type.upper()
    if distance_type not in DISTANCE_MAP:
        raise ValueError(f"Invalid distance_type '{distance_type}'. Must be one of {list(DISTANCE_MAP.keys())}.")
    if not isinstance(k_neighbors, int) or k_neighbors < 1:
        raise ValueError("k_neighbors must be a positive integer.")
    return KNeighborsClassifier(n_neighbors=k_neighbors, metric=DISTANCE_MAP[distance_type])

if len(sys.argv) != 3:
    print(f"Usage: python {sys.argv[0]} <distance_type> <k_neighbors>")
    sys.exit(1)

distance_type = sys.argv[1]
k_neighbors = int(sys.argv[2])

X_train = read_csv_float('csvs/X_train.csv')
X_test = read_csv_float('csvs/X_test.csv')
y_train = read_csv_int('csvs/y_train.csv')
y_test = read_csv_int('csvs/y_test.csv')

start = time.time()
knn = create_knn(distance_type, k_neighbors)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
end = time.time()

print(f"Python KNN time: {end-start:.4f}s")
score = knn.score(X_test, y_test)
print(f"Accuracy = {score:.3f}")


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import csv


n_samples = 40000
n_features = 10
n_classes = 9


X, y = make_classification(
        n_samples = n_samples,
        n_features = n_features,
        n_informative = n_features,
        n_redundant = 0,
        n_classes = n_classes,
        random_state = 42
)


X = X.astype(np.float32)
y = y.astype(np.int32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def write_csv(filename, array, classification):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        if classification:
            for value in array:
                writer.writerow([value])
            return
        for row in array:
            writer.writerow(row)

write_csv('X_train.csv', X_train, False)
write_csv('X_test.csv', X_test, False)
write_csv('y_train.csv', y_train, True)
write_csv('y_test.csv', y_test, True)



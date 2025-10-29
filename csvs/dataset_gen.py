from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import csv

data = load_iris()
X = data.data.astype(np.float32)  
y = data.target.astype(np.int32) 

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

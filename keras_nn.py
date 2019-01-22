import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist

from keras.layers.core import Dense, Activation
from keras.models import Sequential
from keras.optimizers import Adam

def create_model():
    model = Sequential()
    model.add(Dense(784, input_dim=784 ,activation='sigmoid'))
    model.add(Dense(200, activation='sigmoid'))
    model.add(Dense(10, activation='sigmoid'))
    return model

def train_model(model, X, y):
    model.compile(loss='mse', optimizer=Adam(lr=1e-3))
    model.fit(X, y)

def load_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X = []
    y = []
    X_t = []
    y_t = []

    for image,label in zip(X_train, y_train):
        image = np.array(image)
        image = image.flatten()
        _input = (np.asfarray(image) / 255.0 * 0.99) + 0.01
        X.append(_input)
        target = np.zeros(10) + 0.01
        target[int(label)] = 0.99
        y.append(target)

    for image,label in zip(X_test, y_test):
        image = np.array(image)
        image = image.flatten()
        _input = (np.asfarray(image) / 255.0 * 0.99) + 0.01
        X_t.append(_input)
        target = np.zeros(10) + 0.01
        target[int(label)] = 0.99
        y_t.append(target)
    return np.asarray(X), np.asarray(y), np.asarray(X_t), np.asarray(y_t)

X, y, X_t, y_t = load_data()

model = create_model()
train_model(model, X, y)
print(X_t[0].shape)
predict = model.predict(X_t)
print(np.argmax(predict[0]))

(X_train, y_train), (X_test, y_test) = mnist.load_data()

plt.imshow(X_test[0], cmap='gray')
plt.title('Predict label: ' + str(np.argmax(predict[0])) + ' --- ' + 'Actual label: ' + str(y_test[0]))
plt.show()

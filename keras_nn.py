import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist

from keras.layers.core import Dense, Activation
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from pathlib import Path

def create_model():
    model = Sequential()
    model.add(Dense(784, input_dim=784 ,activation='sigmoid'))
    model.add(Dense(200, activation='sigmoid'))
    model.add(Dense(10, activation='sigmoid'))
    return model

def train_model(model, X, y):
    model.compile(loss='mse', optimizer=Adam(lr=1e-3))
    model.fit(X, y)

def scale_data(X_train, y_train,X_test, y_test):
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

def show_predict(predict_label, actual_label,actual_image):
    plt.imshow(actual_image, cmap='gray')
    plt.title('Predict label: ' + str(predict_label) + ' --- ' + 'Actual label: ' + str(actual_label))
    plt.show()

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X, y, X_t, y_t = scale_data(X_train, y_train,X_test, y_test)
    model = Path("model.h5")
    if not model.is_file():
        model = create_model()
        train_model(model, X, y)
        model.save('model.h5')
    else: 
        model = load_model('model.h5')

    predict = model.predict(X_t)
    index_test = 899
    print(np.argmax(predict[index_test]))
    
    show_predict(np.argmax(predict[index_test]),y_test[index_test], X_test[index_test])

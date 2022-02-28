import os

from keras import models
from keras import layers
from keras import optimizers
import numpy as np
from matplotlib import pyplot as plt

class CNN:
    def __init__(self):
        self.model = models.Sequential()
        self.history = []

    def build(self, input_shape, classifiers):
        self.model.add(
            layers.Conv2D(32, (1, 6), strides=(1, 2), activation='relu', input_shape=input_shape, padding='SAME'))

        self.model.add(layers.MaxPooling2D((1, 2), strides=(1, 2)))

        self.model.add(layers.Conv2D(64, (1, 3), strides=(1, 1), activation='relu', padding='SAME'))

        self.model.add(layers.Conv2D(128, (1, 3), strides=(1, 1), activation='relu', padding='SAME'))

        self.model.add(layers.MaxPooling2D((1, 2), strides=(1, 2)))

        self.model.add(layers.Conv2D(128, (6, 1), strides=(1, 1), padding='VALID', activation='relu'))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(classifiers, activation='softmax'))

        self.model.compile(optimizer=optimizers.adam_v2.Adam(lr=0.001), loss='categorical_crossentropy',
                           metrics=['acc'])

    def run(self, epochs, partial_x_train, partial_y_train, x_val, y_val, save_path, loop=1, is_draw=True,
            batch_size=128):
        for n in range(loop):
            self.history.append(self.model.fit(partial_x_train, partial_y_train, epochs=epochs, batch_size=128,
                                               validation_data=(x_val, y_val)))
            # save model:
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            i = 1
            try:
                f = None
                while True:
                    f = open(os.path.join(save_path, 'cnn_{}.h5'.format(i)))
                    i += 1
            except Exception:
                self.model.save(os.path.join(save_path, './cnn_{}.h5'.format(i)))
                print(f"the model is saved as cnn_{i}.h5")
            finally:
                if f is not None:
                    f.close()
            # paint acc-loss
            if is_draw:
                acc = self.history[n].history['acc']
                val_acc = self.history[n].history['val_acc']
                loss = self.history[n].history['loss']
                val_loss = self.history[n].history['val_loss']

                epochs = range(1, len(acc) + 1)

                plt.plot(epochs, acc, 'bo', label='Training acc')
                plt.plot(epochs, val_acc, 'b', label='Validation acc')
                plt.legend()

                plt.figure()
                plt.plot(epochs, loss, 'bo', label='training loss')
                plt.plot(epochs, val_loss, 'b', label='validation loss')
                plt.legend()

                plt.show()

        return self.history

    def get_model(self):
        return self.model

    @staticmethod
    def find_best_fit_model(x_test, y_test, model_path):
        i = 1
        pos=0
        max_acc = 0
        result_model = None
        try:
            f = None
            while True:
                f = open(os.path.join(model_path, 'cnn_{}.h5'.format(i)))
                result_model = models.load_model(os.path.join(model_path, 'cnn_{}.h5'.format(i)))
                loss, acc = result_model.evaluate(x_test, y_test)
                if max_acc < acc:
                    max_acc = acc
                    pos=i
                i += 1
        except Exception:
            if i == 1:
                print('no models are found')
            else:
                print(f"best fit model: cnn_{pos}.h5")
        finally:
            if f is not None:
                f.close()

        return result_model, max_acc

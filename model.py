from keras import losses
from tensorflow import keras


class OriginalModel():
    def __init__(self, loader):
        train_set = loader.train_ds.shuffle(1000)
        self.train_set = train_set
        self.valid_set = loader.val_ds
        self.test_set = loader.test_ds

        self.model = keras.Sequential([
            keras.layers.Rescaling(1. / 255),
            keras.layers.Conv2D(32, 3, activation='relu', padding="same"),
            keras.layers.Dropout(.2),
            keras.layers.Conv2D(32, 3, activation='relu', padding="same"),
            keras.layers.MaxPooling2D(2),
            keras.layers.Conv2D(64, 3, activation='relu', padding="same"),
            keras.layers.Dropout(.2),
            keras.layers.Conv2D(64, 3, activation='relu', padding="same"),
            keras.layers.MaxPooling2D(2),
            keras.layers.Conv2D(128, 3, activation='relu', padding="same"),
            keras.layers.Dropout(.2),
            keras.layers.Conv2D(128, 3, activation='relu', padding="same"),
            keras.layers.MaxPooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(loader.nb_classes, activation='softmax')
        ])

    def train_model(self):

        self.model.compile(
            optimizer='adam',
            loss=losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

        history = self.model.fit(
            self.train_set,
            validation_data=self.valid_set,
            epochs=5,
            batch_size=32
        )
        return history

    def save_model(self,filename):
        self.model.save(filename)

    def predict(self, x):
        y_pred = self.model.predict(x)
        return y_pred



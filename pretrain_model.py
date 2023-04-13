import tensorflow as tf
from tensorflow import keras


def preprocess(image, label):
    resized_image = tf.image.resize(image, [224, 224])
    final_image = keras.applications.xception.preprocess_input(resized_image)
    return final_image, label


class PretrainedModel:
    def __init__(self, loader):
        train_set = loader.train_ds.shuffle(1000)
        self.train_set = train_set.map(preprocess)
        self.valid_set = loader.val_ds.map(preprocess)
        self.test_set = loader.test_ds.map(preprocess)
        self.base_model = keras.applications.xception.Xception(weights="imagenet",
                                                               include_top=False)

        avg = keras.layers.GlobalAveragePooling2D()(self.base_model.output)
        output = keras.layers.Dense(loader.nb_classes, activation="softmax")(avg)
        self.model = keras.Model(inputs=self.base_model.input, outputs=output)

    def model_train(self):
        for layer in self.base_model.layers:
            layer.trainable = False

        optimizer = keras.optimizers.legacy.SGD(learning_rate=0.2, momentum=0.9, decay=0.01)
        self.model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
                           metrics=["accuracy"])

        history = self.model.fit(self.train_set, epochs=3, validation_data=self.valid_set, batch_size=32)
        return history

    def model_train_base(self):
        for layer in self.base_model.layers:
            layer.trainable = True

        optimizer = keras.optimizers.legacy.SGD(learning_rate=0.01, momentum=0.9, decay=0.001)
        self.model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
                           metrics=["accuracy"])

        history = self.model.fit(self.train_set, epochs=3, validation_data=self.valid_set)
        return history

    def predict(self, x):
        y_pred = self.model.predict(x)
        return y_pred

    def save(self, filename):
        self.model.save(filename)

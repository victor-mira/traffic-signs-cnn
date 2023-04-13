from tensorflow import keras


class Loader:
    def __init__(self, data_dir, batch_size, img_height, img_width, train_split, val_split, ds_size):
        self.data_dir = data_dir
        self.img_height = img_height
        self.img_width = img_width
        self.train_size = int(train_split * ds_size)
        self.val_size = int(val_split * ds_size)
        self.test_size = int(val_split * ds_size)
        self.ds_size = ds_size
        self.batch_size = batch_size

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.images_ds = None
        self.nb_classes = None
        self.class_names = None

    def load_for_train(self):
        self.images_ds = keras.utils.image_dataset_from_directory(
            self.data_dir,
            seed=667,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)

        self.class_names = self.images_ds.class_names
        self.nb_classes = len(self.class_names) + 1

        self.train_ds = self.images_ds.take(self.train_size)
        self.val_ds = self.images_ds.skip(self.train_size).take(self.val_size)
        self.test_ds = self.images_ds.skip(self.train_size).skip(self.val_size)

    def load_for_evaluate_and_predict(self):
        self.images_ds = keras.utils.image_dataset_from_directory(
            self.data_dir,
            seed=667,
            image_size=(self.img_height, self.img_width),
            batch_size=None)

        self.class_names = self.images_ds.class_names
        self.nb_classes = len(self.class_names)

        self.train_ds = self.images_ds.take(self.train_size)
        self.val_ds = self.images_ds.skip(self.train_size).take(self.val_size)
        self.test_ds = self.images_ds.skip(self.train_size).skip(self.val_size)

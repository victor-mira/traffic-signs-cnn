from tensorflow import keras
from tensorflow import losses

data_dir = 'traffic_Data/DATA'

batch_size = 32
img_height = 180
img_width = 180

train_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    subset='training',
    validation_split=0.2,
    seed=667,
    image_size=(img_height, img_width),
    batch_size=batch_size)

test_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    subset='validation',
    validation_split=0.2,
    seed=667,
    image_size=(img_height, img_width),
    batch_size=batch_size)


class_names = train_ds.class_names
print(class_names)

num_classes = len(class_names) + 1


model = keras.Sequential([
    keras.layers.Rescaling(1. / 255),
    keras.layers.Conv2D(32, 3, activation='relu'),
    keras.layers.Dropout(.2),
    keras.layers.Conv2D(32, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(32, 3, activation='relu'),
    keras.layers.Dropout(.2),
    keras.layers.Conv2D(32, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
  optimizer='adam',
  loss=losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model.fit(
  train_ds,
  validation_data=test_ds,
  epochs=5
)

model.save('model1')

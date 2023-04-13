import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.preprocessing import label_binarize

from loader import Loader

for i in range(5):
    loader = Loader(data_dir='traffic_Data/DATA', batch_size=32, img_height=180, img_width=180, train_split=0.8,
                    val_split=0.1, ds_size=4170)
    loader.load_for_evaluate_and_predict()
    test_ds = loader.train_ds.batch(32)

    model = tf.keras.models.load_model("original_model_" + str(i))

    results = model.evaluate(test_ds)
    print("original model " + str(i) + " :test loss, test acc:", results)

    model = tf.keras.models.load_model("pretrain_model_" + str(i))

    results = model.evaluate(test_ds)
    print("pretrained model " + str(i) + " :test loss, test acc:", results)


# concatenated_values = tf.concat([element[0] for element in test_ds], axis=0)
# concatenated_label = tf.concat([element[1] for element in test_ds], axis=0)
# y_pred = model.predict(concatenated_values, batch_size=32)
# print("prediction", y_pred)
# y_pred = y_pred.round().astype(int)
# y_pred = np.delete(y_pred, 58,1)
# intClassName = list(map(int, loader.class_names))
#
# y_multiclass = label_binarize(concatenated_label, classes=range(58))
#
# print(classification_report(y_multiclass, y_pred))


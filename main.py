import matplotlib.pyplot as plt

from loader import Loader
from model import OriginalModel
from pretrain_model import PretrainedModel

nb_test = 5

loader = Loader(data_dir='traffic_Data/DATA', batch_size=32, img_height=180, img_width=180, train_split=0.8,
                val_split=0.1, ds_size=4170)
loader.load_for_train()

for i in range(nb_test):
    original_model = OriginalModel(loader)
    history = original_model.train_model()

    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('original_model_accuracies_' + str(i))
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('original_model_losses_' + str(i))
    plt.show()

    original_model.save_model('original_model_' + str(i))

    pretrained_model = PretrainedModel(loader)
    history = pretrained_model.model_train()
    pretrained_model.save("pretrain_model_" + str(i))
    plt.plot(history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('pretrained_model_accuracies_' + str(i))
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('pretrained_model_losses_' + str(i))
    plt.show()

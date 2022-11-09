from os import PathLike
from typing import Union

from importlib_resources import files, as_file
from keras import Model, Input, regularizers
from keras.layers import GRU, Masking, RepeatVector
from keras.losses import mean_squared_error
from keras.optimizers import Adam

from .model_abstract import ModelAbstract


class DSMAutoencoder(ModelAbstract):
    """
    TODO
    """

    def __init__(self, result_dir: Union[str, 'PathLike[str]'] = None):
        super(DSMAutoencoder, self).__init__(result_dir)

    def compile(self, input_dim=6, seq_max_len=2000, hidden_dim=32, optimizer=Adam(0.001, decay=0.001, clipnorm=1.0)):
        """
        TODO

        :param seq_max_len:
        :param hidden_dim:
        :param input_dim:
        :param optimizer:
        :return:
        """
        # compile model
        input = Input((seq_max_len, input_dim,))
        x = Masking(mask_value=0.0)(input)
        x = GRU(128, kernel_regularizer=regularizers.l2(0.001), dropout=0.5, return_sequences=True)(x)
        x = GRU(64, kernel_regularizer=regularizers.l2(0.001), dropout=0.5, return_sequences=True)(x)

        encoder = GRU(hidden_dim, kernel_regularizer=regularizers.l2(0.001),
                      dropout=0.5, return_sequences=False, name='encoder')(x)

        # x = BatchNormalization()(unsupervised)
        x = RepeatVector(seq_max_len)(encoder)
        x = GRU(64, kernel_regularizer=regularizers.l2(0.001), dropout=0.5, return_sequences=True)(x)
        x = GRU(128, kernel_regularizer=regularizers.l2(0.001), dropout=0.5, return_sequences=True)(x)

        decoder = GRU(input_dim, kernel_regularizer=regularizers.l2(0.001), dropout=0.5, return_sequences=True)(x)

        self._model = Model(inputs=input, outputs=decoder)
        self._model.summary()
        self._model.compile(optimizer=optimizer, loss=mean_squared_error)

    def fit(self, data_train, data_test, batch_size=64, epochs=300,
            model_save_path: Union[str, 'PathLike[str]'] = None, shuffle=False):
        """
        TODO

        :param data_train:
        :param data_test:
        :param batch_size:
        :param epochs:
        :param model_save_path:
        :param shuffle:
        :return:
        """

        super(DSMAutoencoder, self)._fit(data_train, data_train, data_test, data_test,
                                         batch_size, epochs, model_save_path, 'val_loss', shuffle)

    def plot_learning_curve(self, path: Union[str, 'PathLike[str]'] = None, fig_size=(20, 20), dpi=200):
        """
        TODO

        :param path:
        :param fig_size:
        :param dpi:
        :return:
        """

        return super(DSMAutoencoder, self)._plot_learning_curve(path, fig_size, dpi, ['loss', 'val_loss'])

    def predict(self, x):
        encoder = Model(inputs=self._model.input, outputs=self._model.get_layer('encoder').output)
        return encoder.predict(x)

    def evaluate(self, x):
        return super(DSMAutoencoder, self)._evaluate(x, x)

    @staticmethod
    def load_1282_seu():
        model = DSMAutoencoder()
        model.compile()
        with as_file(files('dsmtools.models') / 'AE_1282_SEU.h5') as path:
            model.load(path)
        return model

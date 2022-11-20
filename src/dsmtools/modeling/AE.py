from os import PathLike
from typing import Union

from importlib_resources import files, as_file
from keras import Model, Input, regularizers
from keras.layers import GRU, Masking, RepeatVector
from keras.losses import mean_squared_error
from keras.optimizers import Adam
import numpy as np

from .model_abstract import ModelAbstract


class DSMAutoencoder(ModelAbstract):
    """
    The class to compile an autoencoder, train/load model parameters and use it to encode your neuron sequences,
    based on keras API.
    Detailed explanations can be found in [the above guide](#deep-sequential-models).

    To use this class, you can start by invoking `DSMAutoencoder.load_1282_seu` to get our trained model and encode
    your neuron sequences, provided that they are converted to the shape that this model accepts (same as the default
    parameters of the `DSMDataConverter.convert_for_ae`). And then use `DSMAutoencoder.predict` to get the latent
    variables.

    To use this class from the beginning, after initialization, use `DSMAutoencoder.compile`
    to set up a new model, whose dimension should be consistent with your input.
    If you'd like to train your own model, use `DSMAutoencoder.fit`, and `DSMAutoencoder.load` to load a saved one.
    Use `DSMAutoencoder.evaluate` for further evaluation.

    The training history and the model are available as `DSMAutoencoder.history` and `DSMAutoencoder.model`. If you
    are familiar with keras, you know what to do with them, but we offer some basic functions to plot learning stats and
    save the model parameters. Their paths can be specified in each method.
    """

    def compile(self, seq_len=2000, feature_dim=6, latent_dim=32,
                optimizer=Adam(learning_rate=0.001, decay=0.001, clipnorm=1.0)):
        """Compile the AE model with specified parameters, before you can train, load or predict.

        The dimensions of sequence length and feature should be consistent with those in the data conversion,
        used by `DSMDataConverter.convert_for_ae`. The dimension of the latent space is dimension of the encoder output.
        They should also be the same with a trained model, if used, and the default parameters are used by our
        prepared model.

        :param seq_len: The length of the sequence, the second dimension of the input array.
        :param feature_dim: The number of features, the last dimension of the input array.
        :param latent_dim: The dimension of the encoded vector.
        :param optimizer: A keras optimizer with learning parameters.
        """

        input = Input((seq_len, feature_dim,))
        x = Masking(mask_value=0.0)(input)
        x = GRU(128, kernel_regularizer=regularizers.l2(0.001), dropout=0.5, return_sequences=True)(x)
        x = GRU(64, kernel_regularizer=regularizers.l2(0.001), dropout=0.5, return_sequences=True)(x)

        encoder = GRU(latent_dim, kernel_regularizer=regularizers.l2(0.001),
                      dropout=0.5, return_sequences=False, name='encoder')(x)   # return sequence is false here

        x = RepeatVector(seq_len)(encoder)
        x = GRU(64, kernel_regularizer=regularizers.l2(0.001), dropout=0.5, return_sequences=True)(x)
        x = GRU(128, kernel_regularizer=regularizers.l2(0.001), dropout=0.5, return_sequences=True)(x)

        decoder = GRU(feature_dim, kernel_regularizer=regularizers.l2(0.001), dropout=0.5, return_sequences=True)(x)

        self._model = Model(inputs=input, outputs=decoder)
        self._model.summary()
        self._model.compile(optimizer=optimizer, loss=mean_squared_error)

    def load(self, path: Union[str, 'PathLike[str]']):
        super(DSMAutoencoder, self).load(path)

    def fit(self, data_train, data_test, batch_size=64, epochs=300,
            model_save_path: Union[str, 'PathLike[str]'] = None, shuffle=False):
        """
        Model training. Since AE is unsupervised, you only need to provide the input data, for train and test.

        If a model save path is specified, the model will be saved when improved by loss evaluation. This save path
        can also be formatted as a keras model path. Check keras documentation to write the format string.

        :param data_train: An input array for training, with shape of (#neuron, #seq_len, #feature).
        :param data_test: An input array for testing, with shape of (#neuron, #seq_len, #feature).
        :param batch_size: The training batch size.
        :param epochs: The number of training epochs.
        :param model_save_path: The path to save the model by checkpoint.
        :param shuffle: Whether to shuffle the input data.
        """

        super(DSMAutoencoder, self)._fit(data_train, data_train, data_test, data_test,
                                         batch_size, epochs, model_save_path, 'val_loss', shuffle)

    def plot_learning_curve(self, path: Union[str, 'PathLike[str]'] = None, fig_size=(20, 20), dpi=200):
        """Plot the learning curve using the history data after training.

        It will plot a new figure and return the matplotlib handles.
        If a path is specified, the figure will be saved.

        :param path: The save path of the figure.
        :param fig_size: The size of the figure.
        :param dpi: resolution setting.
        :return: The plotted figure's handles, figure and axes.
        """

        return super(DSMAutoencoder, self)._plot_learning_curve(path, fig_size, dpi, ['loss', 'val_loss'])

    def predict(self, x: np.array):
        """Give the encoder's output with given data.

        This will temporarily build a new model using the parameters from the whole model. For efficiency, predicting
        everything in a single shot is preferred.

        :param x: Data to encode, with shape of (#neuron, #seq_len, #feature).
        :return: The predicted latent variables, with shape of (#neuron, #latent_dim).
        """
        encoder = Model(inputs=self._model.input, outputs=self._model.get_layer('encoder').output)
        return encoder.predict(x)

    def evaluate(self, x):
        """Test the model with some input data. As this model is unsupervised, it only needs the input data.
        Your input array should share its dimensions other than the number of neurons with the model.

        :param x: An array for evaluation, with shape of (#neuron, #seq_len, #feature).
        :return: The loss of evaluation.
        """
        return super(DSMAutoencoder, self)._evaluate(x, x)

    @staticmethod
    def load_1282_seu():
        """A static method to build and load an already trained model on 1282 neurons from Southeast University.

        :return: A `DSMAutoencoder` instance with the trained model.
        """
        model = DSMAutoencoder()
        model.compile()
        with as_file(files('dsmtools.modeling') / 'AE_1282_SEU.h5') as path:
            model.load(path)
        return model

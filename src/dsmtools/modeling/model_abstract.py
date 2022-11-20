import os
from os import PathLike
from pathlib import Path
from typing import Union, Optional, Iterable

import matplotlib.pyplot as plt
import numpy as np
from keras import Model
from keras.callbacks import ModelCheckpoint


class ModelAbstract:
    """
    This is an abstract class for the common methods of our deep learning models. It cannot be used directly.
    """

    def __init__(self):
        self._history = None
        self._model: Optional[Model] = None

    @property
    def model(self):
        """The network model. After initialization, it's None, so it should be compiled first."""
        return self._model

    @property
    def history(self):
        """The training history if the model is trained, otherwise is None."""
        return self._history

    def _fit(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,
             batch_size: int, epochs: int, model_save_path: Union[str, 'PathLike[str]'], monitor: str, shuffle: bool):
        callbacks_list = []
        if model_save_path is not None:
            path = Path(model_save_path)
            if not path.parent.exists():
                os.makedirs(path, exist_ok=True)
            callbacks_list.append(
                ModelCheckpoint(
                    filepath=model_save_path,
                    monitor=monitor,
                    save_best_only=True,
                )
            )
        self._history = self._model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                                        callbacks=callbacks_list,
                                        validation_data=(x_test, y_test), shuffle=shuffle)

    def _plot_learning_curve(self, path: Union[str, 'PathLike[str]'], fig_size: tuple[int, int],
                             dpi: int, track: Iterable[str]):
        f, ax = plt.subplots(1, 1, figsize=fig_size)
        track = list(track)
        for t in track:
            ax.plot(self.history.history[t], label=t)
        ax.legend(track)
        if path is not None:
            path = Path(path)
            if not path.parent.exists():
                os.makedirs(path, exist_ok=True)
            plt.savefig(path, dpi=dpi)
        return f, ax

    def load(self, path: Union[str, 'PathLike[str]']):
        """After compiling the model, you can use this method to load a trained model, and their structure must be the
        same.

        :param path: The path to the loaded model.
        """

        self._model.load_weights(path)

    def _evaluate(self, x: np.ndarray, y: np.ndarray):
        return self._model.evaluate(x, y)

    def predict(self, x: np.ndarray):
        """Make prediction using the model, generating numeric results, which may need further conversion for you
        application.

        :param x: Input array of data.
        :return: Prediction array.
        """

        return self._model.predict(x)

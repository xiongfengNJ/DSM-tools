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
    TODO
    """

    def __init__(self, result_dir: Union[str, 'PathLike[str]']):
        """
        TODO

        :param result_dir:
        """

        self._history = None
        self._model: Optional[Model] = None
        self._res_dir: Optional[Path] = None
        if result_dir is not None:
            self._res_dir = Path(result_dir)

    @property
    def model(self):
        return self._model

    @property
    def history(self):
        return self._history

    @property
    def result_dir(self):
        return self._res_dir

    @result_dir.setter
    def result_dir(self, result_dir: Optional[Union[str, 'PathLike[str]']]):
        self._res_dir = None
        if result_dir is not None:
            self._res_dir = Path(result_dir)

    def _fit(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,
             batch_size: int, epochs: int, model_save_path: Union[str, 'PathLike[str]'], monitor: str, shuffle: bool):
        callbacks_list = []
        if model_save_path is not None:
            path = Path(model_save_path)
            if self._res_dir is not None and not path.is_absolute():
                path = str(self.result_dir / path)
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
            if self._res_dir is not None and not path.is_absolute():
                path = self._res_dir / path
            if not path.parent.exists():
                os.makedirs(path, exist_ok=True)
            plt.savefig(path, dpi=dpi)
        return f, ax

    def load(self, path: Union[str, 'PathLike[str]']):
        """
        TODO

        :param path:
        :return:
        """

        self._model.load_weights(path)

    def _evaluate(self, x: np.ndarray, y: np.ndarray):
        return self._model.evaluate(x, y)

    def predict(self, x: np.ndarray):
        """
        TODO

        :param x:
        :return:
        """

        return self._model.predict(x)

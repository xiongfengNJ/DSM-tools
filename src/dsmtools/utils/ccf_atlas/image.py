import SimpleITK as sitk
import numpy as np
import pickle


class Image:
    """
    This class is originally defined to load CCF atlas (value of a pixel is the ID of a brain structure)
    """

    def __init__(self, file=None, pickle_file=None):
        assert ((file is not None) | (pickle_file is not None)), "Error in initializing image object."
        if pickle_file is None:
            self.file = file
            self.image = sitk.ReadImage(file)
        else:
            '''
            To be deprecated: Save large image files as numpy array takes to much space
            '''
            self.file = "Given image"
            array, spacing = pickle.load(
                open(pickle_file, 'rb'))  # Note that pickle does not support SimpleITK image class
            array = np.swapaxes(array, 0, 2)
            self.image = sitk.GetImageFromArray(array)
            self.image.SetSpacing(spacing)
        self.array = sitk.GetArrayViewFromImage(self.image)
        self.array = np.swapaxes(self.array, 0, 2)  # Swap XZ axis to meet the convention of V3D
        # self.values = np.unique(self.array).tolist()
        self.axes = ["x", "y", "z"]
        self.size = dict(zip(self.axes, self.image.GetSize()))
        self.space = dict(zip(self.axes, self.image.GetSpacing()))
        self.micron_size = dict(zip(self.axes,
                                    [self.size["x"] * self.space["x"],
                                     self.size["y"] * self.space["y"],
                                     self.size["z"] * self.space["z"],
                                     ]
                                    ))

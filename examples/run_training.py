from pathlib import Path
import pickle
import numpy as np

from dsmtools import modeling
from sklearn.model_selection import ShuffleSplit

if __name__ == '__main__':
    ds_path = Path('output/dataset.pickle')
    with open(ds_path, 'rb') as f:
        ds = pickle.load(f)
    converter = modeling.DSMDataConverter(ds)
    s = ShuffleSplit(n_splits=2, test_size=0.2, random_state=0)

    # AE training
    ae_x = converter.convert_for_ae()
    train_index, test_index = next(s.split(ae_x))
    ae = modeling.DSMAutoencoder(result_dir='output')
    ae.compile()
    ae.fit(ae_x[train_index], ae_x[test_index], model_save_path='ae_checkpoint.h5', epochs=5)
    ae.evaluate(ae_x[test_index])
    ae.plot_learning_curve('ae_learning.png')

    # HAN training
    label = {r'..\tests\data\swc\17300_6010_x23413_y13446.swc': 'IT_VIS',
             r'..\tests\data\swc\17302_00001.swc': 'CP_SNr',
             r'..\tests\data\swc\17302_00057.swc': 'CP_GPe',
             r'..\tests\data\swc\17302_00112.swc': 'IT_SS',
             r'..\tests\data\swc\17545_00093.swc': 'LGd',
             r'..\tests\data\swc\17545_00094.swc': 'LGd',
             r'..\tests\data\swc\17545_00095.swc': 'IT_SS',
             r'..\tests\data\swc\17545_00166.swc': 'MG',
             r'..\tests\data\swc\17545_00168.swc': 'IT_SS',
             r'..\tests\data\swc\17545_00169.swc': 'IT_SS'}
    han_x, han_y, le = converter.convert_for_han(labels=label)
    train_index, test_index = next(s.split(han_x, han_y))
    han = modeling.DSMHierarchicalAttentionNetwork(result_dir='output')
    han.compile(class_num=len(le.classes_))
    han.fit(han_x[train_index], han_y[train_index], han_x[test_index], han_y[test_index],
            model_save_path='han_checkpoint.h5', epochs=5)
    han.evaluate(han_x[test_index], han_y[test_index])
    han.plot_learning_curve('han_learning.png')

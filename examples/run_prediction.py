from pathlib import Path
import pickle
import numpy as np

from dsmtools import modeling


if __name__ == '__main__':
    ds_path = Path('output/dataset.pickle')
    with open(ds_path, 'rb') as f:
        ds = pickle.load(f)
    converter = modeling.DSMDataConverter(ds)

    ae_x = converter.convert_for_ae()
    ae = modeling.DSMAutoencoder.load_1282_seu()
    print("Autoencoder predictions:")
    for k, v in zip(ds.keys(), ae.predict(ae_x)):
        print(k)
        print(v)

    han_x = converter.convert_for_han()
    han = modeling.DSMHierarchicalAttentionNetwork.load_1282_seu()
    le = modeling.DSMHierarchicalAttentionNetwork.label_encoder_1282_seu()
    print("HAN predictions:")
    for k, v in zip(ds.keys(), le.inverse_transform(np.argmax(han.predict(han_x), axis=1))):
        print(k, v)

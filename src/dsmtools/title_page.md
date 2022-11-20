**DSM-tools** is a Python module that converts neuron morphology into sequences by binary tree
traversals and implements deep learning models for encoding the sequences and predicting cell types.

This project was started by Feng Xiong ([xiongfengNJ](https://github.com/xiongfengNJ)) in 2020, SEU-ALLEN, Nanjing.

The Python package was authorised to be developed by Zuo-Han Zhao ([zzhmark](https://github.com/zzhmark)) in 2022.

<br/>

# Installation

---

## Depedencies
* Python (>=3.9)
* tensorflow (>=2.10.0)
* scikit-learn (>=1.1.3)
* SimpleITK (2.2.0)
* gensim (>=4.2.0)
* matplotlib (>=3.6.2)
* pandas (>=1.5.1)
* importlib_resources (>=5.10.0)

## Install by PyPI
```shell
pip install DSM-tools
```

## Install by GitHub
```shell
pip install git+https://github.com/xiongfengNJ/DSM-tools
```

<br/>

# User Guide

---

Here's some simple usage that get you a quick start.

## Transform SWC files to sequence dataframes

```python
from dsmtools.preprocessing import NeuronSequenceDataset

# dataset generation tool, with computation settings
ds = NeuronSequenceDataset(swc_file_paths, jobs=10)

# processing
ds.read_parallel()
ds.qc_parallel(qc_len_thr=10)
ds.make_sequence_parallel()

# save the result (OrderedDict) as pickle
ds.pickle_result('output.pickle')
```
By default, it gives you a set of features for each neuron by the order of preorder traversals.

## Predict cell types with HAN model


```python
from dsmtools.modeling import DSMDataConverter, DSMHierarchicalAttentionNetwork

# further convert the dataframes to dataset fed to tensorflow
converter = DSMDataConverter(ds)
han_x = converter.convert_for_han()

# models trained with our data ready for use
han = DSMHierarchicalAttentionNetwork.load_1282_seu()
le = DSMHierarchicalAttentionNetwork.label_encoder_1282_seu()

# decode the one-hot matrix back to labels
le.inverse_transform(np.argmax(han.predict(han_x), axis=1))
```
The prediction for autoencoder is similar.

## Train an autoencoder

```python
from dsmtools.modeling import DSMAutoencoder

ae_x = converter.convert_for_ae()

# build model
ae = DSMAutoencoder(result_dir='output')
ae.compile(feature_dim=6, seq_len=2000)

# training
ae.fit(train_x, test_x, model_save_path='ae_checkpoint.h5', epochs=300, batch_size=32)

ae.plot_learning_curve('ae_learning.png')
```
The training for HAN is similar.

Please see the [examples](https://github.com/xiongfengNJ/DSM-tools/tree/master/examples) directory for details.
## Fine tune the data processing

You can inherit classes like `dsmtools.preprocessing.NeuronSequenceDataset`
to change the data processing behaviours, which should be quite easy.
The tree manipulating class `dsmtools.preprocessing.NeuronTree`
offers you the freedom of exploring your own definition of subtree nodes to generate traversal sequences.

<br/>

# Contact & Support

---

## Documentation

DSM-tools API Documentation: https://xiongfengnj.github.io/DSM-tools
powered by pdoc, served in GitHub Pages.

## GitHub projects

Package source: https://github.com/xiongfengNJ/DSM-tools

Experiment code: https://github.com/xiongfengNJ/neuron2seq

Experiment data: http://101.43.104.173:8500

Online service: http://101.43.104.173:8501

## Team

SEU-ALLEN Joint Center,
Institute for Brain and Intelligence,
Southeast University

Lab homepage: https://braintell.org

<br/>

# Citation

---

Our paper is still under review, please check the link for preprint. For now, you can cite it as:

Hanchuan Peng, Feng Xiong, Peng Xie et al. DSM: Deep Sequential Model for Complete Neuronal Morphology Representation
and Feature Extraction, 29 June 2022, PREPRINT (Version 1) available at Research Square
[https://doi.org/10.21203/rs.3.rs-1627621/v1]
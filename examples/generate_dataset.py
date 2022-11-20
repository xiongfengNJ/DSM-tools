from pathlib import Path

from dsmtools.preprocessing import NeuronSequenceDataset


if __name__ == '__main__':
    swc_dir = Path('../sample_data/swc')
    a = swc_dir.rglob('*.swc')
    ds = NeuronSequenceDataset(a, jobs=10, debug=True)
    ds.read_parallel()
    ds.qc_parallel()
    ds.make_sequence_parallel()
    ds.pickle_result('output/dataset.pickle')

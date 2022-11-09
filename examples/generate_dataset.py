from pathlib import Path

from dsmtools.preprocessing import NeuronSequenceDataset


if __name__ == '__main__':
    swc_dir = Path('../tests/data/swc')
    a = swc_dir.rglob('*.swc')
    ds = NeuronSequenceDataset(a, jobs=10, debug=True)
    ds.read_parallel()
    ds.qc_parallel()
    ds.make_sequences_parallel()
    ds.pickle_sequences('output/dataset.pickle')

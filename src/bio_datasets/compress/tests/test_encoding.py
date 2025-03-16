import numpy as np

from bio_datasets.compress.encoding import HistogramEncoding, SparseHistogramEncoding


def test_sparse_histogram_encoding():
    values = np.zeros(100)
    sparsity_mask = np.random.rand(100) < 0.5
    values[~sparsity_mask] = np.random.randn(100 - np.sum(sparsity_mask))
    encoder = SparseHistogramEncoding.build(
        values, num_bins=1000, offset=0.0, zero_threshold=0.01
    )
    encoded = encoder.encode(values)
    decoded = encoder.decode(encoded)
    print(values)
    print(decoded)

    assert np.allclose(values, decoded, atol=1e-2)


def test_histogram_encoding():
    values = np.zeros(100)
    sparsity_mask = np.random.rand(100) < 0.5
    values[~sparsity_mask] = np.random.randn(100 - np.sum(sparsity_mask))
    encoder = HistogramEncoding.build(values, num_bins=1000, low=None, high=None)
    encoded = encoder.encode(values)
    decoded = encoder.decode(encoded)

    assert np.allclose(values, decoded, atol=1e-2)

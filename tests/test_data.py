from torch.utils.data import Dataset
import torch
import numpy as np
import os.path
import pytest
from pumpkin_spice_cookie.data import corrupt_mnist, normalize

@pytest.mark.skipif(not os.path.exists("data/processed"), reason="Data files not found")
def test_my_dataset():
    """Test the MyDataset class."""
    dataset_train, dataset_test = corrupt_mnist()
    N_train = 30000
    N_test = 5000
    assert isinstance(dataset_train, Dataset)
    assert len(dataset_train) == N_train
    assert np.all([datapoint[0].shape==torch.Size([1,28,28]) for datapoint in dataset_train])

    assert isinstance(dataset_test, Dataset)
    assert len(dataset_test) == N_test
    assert np.all([datapoint[0].shape==torch.Size([1,28,28]) for datapoint in dataset_test]) 

def test_normalize():
    """Test the normalize function."""
    images = torch.randn(10, 1, 28, 28)
    images_normalized = normalize(images)   
    assert torch.allclose(images_normalized.mean(), torch.tensor(0.), atol=1e-2)
    assert torch.allclose(images_normalized.std(), torch.tensor(1.), atol=1e-2)
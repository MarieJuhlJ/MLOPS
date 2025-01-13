from pumpkin_spice_cookie.model import MyAwesomeModel
import torch
import pytest

def test_my_model():
    model = MyAwesomeModel()
    assert model(torch.randn(1, 1, 28, 28)).shape == torch.Size([1, 10])

@pytest.mark.parametrize("batch_size", [32, 64])
def test_model(batch_size: int) -> None:
    model = MyAwesomeModel()
    x = torch.randn(batch_size, 1, 28, 28)
    y = model(x)
    assert y.shape == (batch_size, 10)
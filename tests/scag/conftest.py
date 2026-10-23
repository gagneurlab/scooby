import torch

# Run eagerly: some alphagenome-pytorch builds torch.compile their gelu.
torch._dynamo.config.disable = True


def pytest_configure(config):
    torch.manual_seed(0)

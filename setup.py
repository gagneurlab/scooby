from setuptools import setup, find_packages

setup(
    name='scooby',
    version='0.1.0',
    author='Johannes Hingerl, Laura Martens',
    author_email='',
    packages=find_packages(),
    url='https://github.com/gagneurlab/scooby',
    license='MIT',
    description='scooby: Modeling multi-modal genomic profiles from DNA sequence at single-cell resolution',
    install_requires=[
        "accelerate >= 0.24.1",
        "enformer-pytorch >= 0.8.10",
        "borzoi-pytorch >= 0.4.0",
        "peft @ git+https://github.com/lauradmartens/peft.git",
        "scanpy >= 1.10.3",
        "pybigtools == 0.1.1",
        "pyarrow >= 15.0.0",
        "intervaltree >= 3.1.0",
        "wandb",
    ],
)

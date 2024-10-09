from setuptools import setup, find_packages

setup(
    name='scooby',
    version='0.0.1',
    author='',
    author_email='',
    packages=find_packages(),
    url='https://github.com/gagneurlab/scooby',
    license='LICENSE',
    description='scooby: Modeling multi-modal genomic profiles from DNA sequence at single-cell resolution',
    install_requires=[
        "accelerate >= 0.24.1",
        "enformer-pytorch >= 0.8.9",
        "borzoi-pytorch @ git+https://github.com/johahi/borzoi-pytorch.git",
        "peft @ git+https://github.com/lauradmartens/peft.git",
        "scanpy >= 1.10.3",
        "pybigtools>=0.1.1"
    ],
)

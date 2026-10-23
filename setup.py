from setuptools import setup, find_packages

setup(
    name='scooby',
    version='0.2.0',
    author='Johannes Hingerl, Laura Martens',
    author_email='',
    packages=find_packages(include=["scooby", "scooby.*"]),
    package_data={"scooby.scag": ["configs/*.yaml"]},
    url='https://github.com/gagneurlab/scooby',
    license='MIT',
    description='scooby: Modeling multi-modal genomic profiles from DNA sequence at single-cell resolution',
    # alphagenome-pytorch requires Python 3.12.
    python_requires=">=3.12",
    install_requires=[
            "accelerate >= 1.3.0",
        "enformer-pytorch >= 0.8.10",
        "borzoi-pytorch >= 0.4.0",
        # genomicsxai/alphagenome-pytorch; PyPI releases before 0.3 are a different package.
        "alphagenome-pytorch >= 0.3.1",
        "torch >= 2.1.0",
        "einops >= 0.6.0",
        "safetensors >= 0.4.3",
        "pyyaml >= 6.0",
        "huggingface_hub >= 0.21.0",
        "peft @ git+https://github.com/lauradmartens/peft.git",
        "scanpy >= 1.10.3",
        "pybigtools == 0.1.1",
        "pyarrow >= 15.0.0",
        "intervaltree >= 3.1.0",
        "wandb",
    ],
    extras_require={
        "test": ["pytest"],
    },
)

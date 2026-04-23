from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="Brain2Vec",
    version="1.0.1",
    description="Implementation of Brain2Vec: Deep Learning Radiomic Features for Neuroimaging Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Spiros Maggioros",
    author_email="software@cbica.upenn.edu",
    maintainer="Spiros Maggioros",
    maintainer_email="Spiros.Maggioros@pennmedicine.upenn.edu",
    download_url="https://github.com/CBICA/Brain2Vec/",
    url="https://github.com/CBICA/Brain2Vec/",
    packages=find_packages(exclude=[".github"]),
    python_requires=">=3.9",
    install_requires=[
        "torch",
        "timm",
        "torchvision",
        "torchio",
        "torchmetrics",
        "tqdm",
        "numpy",
        "pandas",
    ],
    entry_points={
        "console_scripts": [
            "Brain2Vec= Brain2Vec.__main__:main",
            "inference=inference:main",
            "svm_train=train_svm:main",
        ]
    },
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    license="By installing/using Brain2Vec, the user agrees to the following license: See https://www.med.upenn.edu/cbica/software-agreement-non-commercial.html",
    keywords=[
        "deep learning",
        "foundation model",
        "medical image analysis",
    ],
    package_data={"Brain2Vec": ["VERSION"]},
)

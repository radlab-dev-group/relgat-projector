from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).parent
readme_path = ROOT / "README.md"
long_description = (
    readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
)

setup(
    name="relgat-projector-trainer",
    version="0.2.1",
    description="A lightweight trainer for frozen-RelGAT with projection layer.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    url="https://github.com/radlab-dev-group/relgat-projector",
    author="RadLab team",
    author_email="pawel@radlab.dev",
    keywords=[
        "graph-neural-networks",
        "GNN",
        "GAT",
        "knowledge-graphs",
        "NLP",
        "PyTorch",
    ],
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    py_modules=["relgat_projector"],
    include_package_data=True,
    install_requires=[
        "numpy",
        "spacy",
        "wandb",
        "accelerate",
        "transformers",
        "sentence-transformers",
        "torch",
        "torchvision",
        "torch-scatter",
    ],
    extras_require={
        "dev": [
            "black",
            "isort",
            "flake8",
        ],
    },
    entry_points={
        "console_scripts": [
            "relgat-projector-train=relgat_projector_apps.trainers.relgat_projector:main",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Documentation": "",
        "Source": "",
        "Tracker": "",
    },
)

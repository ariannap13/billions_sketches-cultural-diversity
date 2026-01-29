# Overview

This repo contains code used in the paper "Billions of Sketches Reveal Hidden Cultural Variation in Human Concepts" by Arianna Pera, Mauro Martino, Nima Dehmamy, Douglas Guilbeault, Luca Maria Aiello, and Andrea Baronchelli.

Each folder contains its specific Readme file. Folder should be approached in the following way:
* ``data_preparation-clustering`` contains code to handle drawings and their embeddings, perform clustering and run some analysis. 
* ``odds_ratios`` contains code related to odds ratios computation.
* ``results`` collects some generated files, and is shared as empty.
* ``useful_functions`` contains some utility code.
* ``word_vs_images_vs_culture`` contains code to perform the comparison between language- and image-based representation of categories, which can lead to a measure of similarity between countries, and the cultural similaritis between countries.

A sample of the data is available from [Google Creative Lab](https://github.com/googlecreativelab/quickdraw-dataset?tab=readme-ov-file).

# System requirements

All code is run in Python version 3.8.16. 
Specific package requirements are listed in ``requirements.txt``. 
The ``dbcv`` library is installed from the [DBCV Github repo](https://github.com/FelSiq/DBCV").

Image embeddings, dimensionality reduction and clustering have been performed on a NVIDIA v100 GPU machine. 

# Installation guide

Follow the steps below to set up the environment and install all dependencies.

## 1. Clone the repository
First, make sure git is installed on your system. You can find more details on the installation process [here](https://github.com/git-guides/install-git).

Clone the repository to your local machine using git:
```
git clone https://github.com/ariannap13/billions_sketches-cultural-diversity.git
```

Then navigate into the project directory:
```
cd billions_sketches-cultural-diversity
```

## 2. Create a virtual environment with Python 3.8.16
Make sure Python 3.8.16 is installed. 

If Python 3.8.16 is not already installed, download and install it from the official [Python website](https://www.python.org/downloads/release/python-3816/).

Make sure to:
- select the correct installer for your operating system
- check the box that says “Add Python to PATH” during installation (especially on Windows)

Then create and activate a virtual environment:
```
python3.8 -m venv venv
```

Activate the virtual environment:

* On macOS/Linux:
    ```
    source venv/bin/activate
    ```
* On Windows:
    ```
    venv\Scripts\activate
    ```

## 3. Install required packages
Install all dependencies listed in `requirements.txt`:

```
pip install --upgrade pip
pip install -r requirements.txt
```

Install ``dbcv`` with 

```
pip install git+https://github.com/FelSiq/DBCV.git
```

The whole installation should take around X minutes to complete.

# Use 

Specific information and instructions to run on data are reported in individual folders, together with descriptions of expected outputs.

The run time for code depends on the amount of data processed. Running image embeddings, dimensionality reduction, and clustering can be computationally intensive.

# Spatially Transformed Adversarial Examples (stAdv) Reproduction

This project aims to reproduce the paper [Spatially Transformed Adversarial Examples](https://arxiv.org/abs/1801.02612),
in which they provide a new direction in adversarial example generation.

Our blog can be found [here](https://medium.com/@henwei1998/8ab4838e448b).

The goal of this project was to reproduce the [Table 1](table1.png), and [Figure 2](figure2.png) of the original paper.

<<<<<<< HEAD
## Installation
To get a local copy up and running follow these steps:

1. **Clone the repository**:\
    Open a terminal and navigate to the location that you want to clone the project in then run the following command:
    ```bash
    git clone https://github.com/erenaydoslu/deep-learning-final-project
    ```
   
2. **Install Python**\
    The code is written in Python.
    Python can be downloaded [here](https://www.python.org/downloads/).

3. **Opening interactive python notebooks (ipynb)**\
    All the code is in this singular interactive python notebook file `sea-pix-GAN.ipynb`, to open such a file you need a special editor.
    The following options were used during development, [Jupyter](https://jupyter.org/), [Google Colab](https://colab.google/), and [VS Code](https://code.visualstudio.com/).


## Running
To run our reproduction first open the notebook file `StAdv_attack.ipynb` in your editor of choice.
Then you can safely run all cells to import the packages (if they don't work insert a cell above and import them using `pip` if installed).
You can either run the code under the caption **Train models**, or you can already go to section **StAdv Attack** where you will get to use the pre-models in the repository. One you can either run adveserial attacks on all digits or one specific digit. Both are under their relative section with the same name.

## Authors
- Eren Aydoslu
- Cem Levi
- Igor Witkowski
- Henwei Zeng 


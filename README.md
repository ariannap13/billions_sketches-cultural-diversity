This repo contains code used in the paper "Billions of Sketches Reveal Hidden Cultural Diversity in Human Concepts" by Arianna Pera, Mauro Martino, Nima Dehmamy, Douglas Guilbeault, Luca Maria Aiello, and Andrea Baronchelli.

Each folder contains its specific Readme file. Folder should be approached in the following way:
* ``data`` contains data used in the code. This folder is shared as empty, as it should host working files produced by the code. A link to a sample of the original Quick, Draw! dataset is shared in the paper.
* ``data_preparation`` contains code to handle drawings and their embeddings, perform clustering and run some analysis. 
* ``odds_ratios`` contains code related to odds ratios computation.
* ``plots_downstream`` collects all generated downstream plots, and is shared as empty.
* ``results`` collects some generated files, and is shared as empty.
* ``useful_functions`` contains some utility code.
* ``word_vs_images_vs_culture`` contains code to perform the comparison between language- and image-based representation of categories, which can lead to a measure of similarity between countries, and the cultural similaritis between countries.

All code is run in 3.8.16. Requirements are listed in ``requirements.txt``. The ``dbcv`` library is installed from the [DBCV Github repo](https://github.com/FelSiq/DBCV").
Code has been run on an HPC cluster managed through a SLURM system. 
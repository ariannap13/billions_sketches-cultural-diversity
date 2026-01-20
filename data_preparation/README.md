# Data preparation and clustering

This folder contains the pipeline to handle a category of drawings, from data preparation to clustering.

The script ``run_category.sh``is the main one to consider, as it automatically calls the jobs needed in sequence for a single category.

All code has been run in Python 3.8.16.

Install the requirements in ``requirements.txt``. Install dbcv module from github[https://github.com/FelSiq/DBCV] (``python -m pip install "git+https://github.com/FelSiq/DBCV"``).

For each category:
1. Choose a category.
2. Place the zipped files of the category's csv files in ``csv_dir`` (``./data/csv_files`` or change the directory accordingly).
3. Place the zipped files of the category's embedding files in ``emb_dir`` (``./data/emb_files`` or change the directory accordingly).
4. Run the script ``run_category.sh`` for the category.

Then, compute the distribution of drawing durations with ``get_time_distribution.py`` and count the number of sketches per country with ``count_countries.job`` (then processed and summarized in ``count_countries.ipynb``).

After clustering is obtained (with code in ``run_category.sh``), ``dbscan_evaluate2d_clusterability.py`` is run to obtain clusterability metrics for the categories and ``perc_noise_clusterability.ipynb`` allows to detect the threshold for the percentage of noise to consider a category as clusterable or non-clusterable. The same code also explores correlations between clusterability and conceptual properties of objects. Specifically, those are obtained by considering existing scores for some of the categories and generating scores with the code in ``./meta-categorization/llama/prompting/`` for the remaining ones.
In order to obtain clean data with id of drawings and cluster of reference for clusterable categories, ``create_files_umap_plots_data.py`` is used.

For "non-clusterable" categories, we run the code in ``./grid_components``(see README there).

We then look at the distribution of the number of clusters in categorie in ``compute_n_clusters.py``.


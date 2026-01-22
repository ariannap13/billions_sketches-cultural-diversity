# Data preparation and clustering

This folder contains the pipeline to handle a category of drawings, from data preparation to clustering.

For each category:
1. Choose a category.
2. Place the zipped files of the category's csv files in ``csv_dir`` (``../data/csv_files`` or change the directory accordingly).
3. Place the zipped files of the category's embedding files in ``emb_dir`` (``../data/emb_files`` or change the directory accordingly).
4. Run the script ``extract_files.py`` by specifying the csv folder to unzip all csv files.
5. Run the script ``duration_drawings.py`` to compute the time required to draw.
6. Run the script ``extract_files.py`` by specifying the embedding folder to unzip all embeddings.
7. Run the script ``merge_embeddings.py`` to merge all embeddings in a single file.
8. By specifying samples_per_country (in our case, 10000), top_countries (100), batch_size (10000), save_every (100000), ndims (2), run ``run_umap.py``, delete unzipped embeddings files with ``delete_files.py`` by specifying the embedding folder and run ``dbscan_run.py``.
9. Run ``compute_cluster_validity.py`` with 20 different seeds to get the optimization metrics for DBSCAN.
10. Delete the unzipped csv files with ``delete_files.py`` by specifying the csv folder.


Then, compute the distribution of drawing durations with ``get_time_distribution.py`` and count the number of sketches per country with ``count_countries.py`` (then processed and summarized in ``count_countries.ipynb``).

After clustering is obtained, ``dbscan_evaluate2d_clusterability.py`` is run to obtain clusterability metrics for the categories and ``perc_noise_clusterability.ipynb`` allows to detect the threshold for the percentage of noise to consider a category as clusterable or non-clusterable. The same code also explores correlations between clusterability and conceptual properties of objects. Specifically, those are obtained by considering existing scores for some of the categories and generating scores with the code in ``./meta-categorization/llama/prompting/`` for the remaining ones. Note that the zero-shot inference with LLama requires an huggingface token to use the inference client.
In order to obtain clean data with ID of drawings and cluster of reference for clusterable categories, ``create_files_umap_plots_data.py`` is used.

For "non-clusterable" categories, we run the code in ``./grid_components`` to perform alternative clustering (see README there).

We then look at the distribution of the number of clusters in categorie in ``compute_n_clusters.py``.


# Compare similarity between words and images

In the comparison between image and word similarities, the following pipeline is used:
1. The similarity between image embeddings of reference categories is computed in ``compute_similaritymatrix_allemb.py``.
2. Multilingual and Word2Vec word embeddings are computed in ``./word-embeddings/multilingual-embed.py`` and ``./word-embeddings/word2vec-embed.py``, respectively.
3. RBO and Kendall-Tau scores for rankings are computed and visualized in ``compare_matrices_rankings.py``.

We then turn to the comparison of image- and word-based network with a culture-based one.
We first define a network of culture:
1. Run ``./culture_network/fix_files.py`` to generate distance matrices by period.
2. Run ``./culture_network/put_files_together`` to create a single file from multiple periods.
3. Run ``./culture_network/culture_similarity_network-clean.py`` to generate the network for cultural similarities between countries. 

Then, we generate the similarity matrix of pairs of countries with ``get_countries_similarities.job` for images. For words, we follow this pipeline:
1. We need to translate the categories names in the language of the top-x countries. We first run ``./word-embeddings/check_countries.py`` to generate a mapping between countries and languages and create empty language files to populate with translated categories.
2. We use Google Translate to do translate in each language, and save the translated words in the corresponding language csv file. 
3. Then, we compute the embeddings in the given language with ``./word-embeddings/multilingual_languages.py``.
4. We generate the multilingual similarity matrix with ``./word-embeddings/multilingual_compare_simmatrix.py`` and we select the relevant subset given a certain number of top countries to consider with ``./word-embeddings/select_emb_sim_matrix.py``.

We can generate image- and words-based networks to visualize through Gephi with ``generate_networks_basics.py``. We can then visualize one network on top of the coordinates of the others by:
1. First, we need to modify the intro in each file generated through Gephi to be readable
    ```json 
    <gexf version="1.2" xmlns="http://www.gexf.net/1.2draft" 
    xmlns:viz="http://www.gexf.net/1.2/viz" 
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
    xsi:schemaLocation="http://www.gexf.net/1.2draft 
    http://www.gexf.net/1.2draft/gexf.xsd">
    ```
2. Run ``open_file.py`` to extract xy coordinates from network files generated through Gephi.
3. Run ``copy_coordinates_fromfile.py`` to take the coordinates of the clusterable categories network as a reference and map all other networks on those, to favor comparison.
4. Run ``open_plot_networkx.py`` to generate final plot of networks.

To compare image, word and similarity networks, we follow this pipeline:
1. Run ``compare_networks-clean.py`` to generate the metrics for similarity between networks.
2. Run ``analysis_culture_comparison.ipynb`` to generate the plots for the comparison between networks.

The file ``generate_networks_basics_100nodes-perc_edges.py`` contains data to generate the comparison between different percentages of strongest edges retained in the network.
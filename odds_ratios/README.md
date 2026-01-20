# Odds Ratios and similarity

The following pipeline is followed:
1. ORs based on cluster assignments and country information are computed in ``compute_ORs_cat_clust.py`` and ``compute_ORs_cat.py``.

These ORs will then be used to compute the similarity for pairs of countries (see ``./word_vs_images_vs_culture/get_countries_similarities.py``).
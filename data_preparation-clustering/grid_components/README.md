# Grid components for non-clusterable categories

The following pipeline is followed:
1. A grid with points count over each UMAP representation of a category is built in ``grid_density_noise.py``.
2. Connected components are obtained through ``grid_create_graph_components.py``.
3. The performance of the process, evaluated over clusterable categories, is obtained in ``grid_components_clust_performance.py``. The performance can then be visualized with ``precision_recall_plot.ipynb``.

To create clean files with drawings ids and cluster of reference, the file ``get_components_as_clusters_data.py`` is used.

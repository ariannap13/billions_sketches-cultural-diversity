import pandas as pd

dimensions = "all"
top_countries = 100

matrix_2010_2014 = pd.read_csv(f"../../data/{dimensions}/{dimensions}_dist_matrix_2010_top{top_countries}.csv", index_col=0)
matrix_2005_2009 = pd.read_csv(f"../../data/{dimensions}/{dimensions}_dist_matrix_2005_top{top_countries}.csv", index_col=0)
matrix_1999_2004 = pd.read_csv(f"../../data/{dimensions}/{dimensions}_dist_matrix_1999_top{top_countries}.csv", index_col=0)
matrix_1994_1998 = pd.read_csv(f"../../data/{dimensions}/{dimensions}_dist_matrix_1994_top{top_countries}.csv", index_col=0)


matrix_ref = matrix_2010_2014.copy()

# Fill only NaN values in the most recent matrix, using older data as backup
matrix_ref = matrix_ref.fillna(matrix_2005_2009)
matrix_ref = matrix_ref.fillna(matrix_1999_2004)
matrix_ref = matrix_ref.fillna(matrix_1994_1998)

# for how many countries do we have data?
n_countries_with_data = (matrix_ref.notna().sum(axis=1) > 0).sum()
print(f"Number of countries with data: {n_countries_with_data} out of {len(matrix_ref.columns)}")

# fill diagonal with 0
for country in matrix_ref.columns:
    matrix_ref.loc[country, country] = 0.0

sim_matrix = 1-matrix_ref
sim_matrix.to_csv(f"../../data/{dimensions}/{dimensions}_sim_matrix_top{top_countries}.csv")




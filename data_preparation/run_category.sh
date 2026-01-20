# to be changed according to category of reference
category="$1"
echo "category=$1"

# home directories for csv files and embeddings, change if needed (avoid the last /)
csv_dir=../data/csv_files # contact the authors for access
emb_dir=../data/emb_files # contact the authors for access
dbscan_save_dir=../data/umap_files

# unzip csv files
sbatch --dependency=singleton --job-name=Group$category unzip.job "$category" "$csv_dir/" "csv"

# drawing duration metrics
sbatch --dependency=singleton --job-name=Group$category duration.job "$category" "$csv_dir/" 

# unzip embedding files
sbatch --dependency=singleton --job-name=Group$category unzip.job "$category" "$emb_dir/" "emb"

# merge embeddings to create single embeddings file
sbatch --dependency=singleton --job-name=Group$category merge.job "$category" "$emb_dir/" 

# get umap embeddings
samples_per_country=10000
top_countries=100
batch_size=10000
save_every=100000
ndims=2
sbatch --dependency=singleton --job-name=Group$category get_umap.job "$category" "$emb_dir/$category-embeddings/$category-embeddings/" "$csv_dir/$category/$category/" "$emb_dir/umap_notime/" $batch_size $save_every $samples_per_country $top_countries $ndims "$emb_dir/"

# delete unzipped embedding files
sbatch --dependency=singleton --job-name=Group$category delete.job "$category" "$emb_dir/"

# run DBSCAN
sbatch --dependency=singleton --job-name=Group$category get_dbscan.job "$category" $ndims "$emb_dir/" "$dbscan_save_dir/"

# run dbcv for DBSCAN
n_seed=20
ndim=2

for seed in $(seq 0 5 $((n_seed-1)))
do
    echo "seed: $seed"
    sbatch --dependency=singleton --job-name=Group$category dbcv.job "$category" "$seed" $ndim "$dbscan_save_dir/" "$emb_dir/" &
done

wait

# # run evaluate all country
# sbatch --dependency=singleton --job-name=Group$category evaluate_countries.job "$category" "$dbscan_save_dir/" "$csv_dir/"

# delete unzipped csv files
sbatch --dependency=singleton --job-name=Group$category delete.job "$category" "$csv_dir/"

# delete .out files with the category name
sbatch --dependency=singleton --job-name=Group$category delete_out.job "$category"
import numpy as np
from glob import glob
import os
import h5py
import torch
import pandas as pd
import umap
# from sklearn.cluster import DBSCAN
# from collections import Counter

import matplotlib.pyplot as plt

V = lambda x: x.detach().cpu().numpy()

def PCA_torch(embeddings, k = 10):
    # Assuming 'embeddings' is already a PyTorch tensor with data on GPU
    # Center the data
    emb_means = embeddings.mean(dim=0)
    centered_data = embeddings - emb_means

    # Perform SVD
    u,s,vh = torch.linalg.svd(centered_data, full_matrices=False)

    # Take the first k principal components
    principal_components = vh[:, :k]

    # Project the centered data onto the principal components to get the PCA scores
    pca_scores = centered_data @ principal_components

    return pca_scores

# we also want a PCA class which can project any input data onto the PCA space of the training data
class PCA:
    def __init__(self, train_data, k=10):
        # Assuming 'train_data' is a PyTorch tensor with data on GPU
        # Center the data
        self.mean = train_data.mean(dim=0)
        centered_data = train_data - self.mean

        # Perform SVD
        u,s,vh = torch.linalg.svd(centered_data, full_matrices=False)

        # Take the first k principal components
        self.principal_components = vh[:, :k]
        
    def project(self, data):
        # Assuming 'data' is a PyTorch tensor with data on GPU
        centered_data = data - self.mean
        return centered_data @ self.principal_components
    

def load_embedding_h5(emb_file, sample_every = 100, verbose=False):
    """
    sample_every: sample from the embedding batch groups and return 1 in every <sample_every>.  
    """
    with h5py.File(emb_file, 'r') as f:
        embeddings = []
        ids = []

        for batch_name in f.keys():
            if verbose:
                print(batch_name)
            batch_group = f[batch_name]
            batch_ids = batch_group['key_id'][:][::sample_every]
            batch_embeddings = batch_group['embeddings'][:][::sample_every]

            embeddings.append(torch.tensor(batch_embeddings))
            ids.extend(batch_ids)

        embeddings_tensor = torch.vstack(embeddings)
    
    return {'key_id': ids, 'embeddings': embeddings_tensor}

# import pandas as pd


def load_embedding_h5_sampled(emb_files, sample_key_ids = None, verbose=False):
    """
    sample_every: sample from the embedding batch groups and return 1 in every <sample_every>.  
    """
    if type(emb_files)==str:
        emb_files = [emb_files]
    # if len(emb_files)>1:
    print(f'Loading from {len(emb_files)} files.')

    embeddings = []
    ids = [] 
    for emb_file in emb_files:
        print(emb_file, end='\r')
        with h5py.File(emb_file, 'r') as f:
            # embeddings = []
            # ids = []
    
            for batch_name in f.keys():
                
                batch_group = f[batch_name]
                batch_ids = batch_group['key_id'][:]
                batch_embeddings = batch_group['embeddings'][:]
                if verbose:
                    print(batch_name, len(batch_embeddings))
    
                # Filter based on sample_key_ids if provided
                if sample_key_ids is not None:
                    filtered_indices = [i for i, key_id in enumerate(batch_ids) if key_id in sample_key_ids]
                    batch_ids = batch_ids[filtered_indices]
                    batch_embeddings = batch_embeddings[filtered_indices]
    
    
                embeddings.append(torch.tensor(batch_embeddings))
                ids.extend(batch_ids)
    
    embeddings_tensor = torch.vstack(embeddings)
    assert len(ids)==len(embeddings_tensor)
    return {'key_id': ids, 'embeddings': embeddings_tensor}


# to sample batches we define a class for loading embeddings
class EmbeddingLoader:
    def __init__(self,emb_files):
        self.prep_emb(emb_files)
        self.emb_file_iterator = iter(self.emb_files)    

    def prep_emb(self,emb_files):
        if type(emb_files)==str:
            emb_files = [emb_files]
        # if len(emb_files)>1:
        print(f'Loading from {len(emb_files)} files.')
        self.emb_files = emb_files

    def open_next(self):    
        self.file_name = next(self.emb_file_iterator)
        self.file_obj = h5py.File(self.file_name, 'r')
        # we need something similar to the load_embedding_h5_sampled function here.
        # the keys in file_ob are the batch names, so we need to iterate over them.
        # when we open a new file we need to reset the iterator
        self.batch_iterator = iter(self.file_obj)
        self.batch_key = next(self.batch_iterator)
        self.batch_group = self.file_obj[self.batch_key]
        self.batch_idx = 0

    # we want to get batches of fixed size until the file is finished.
    def get_batch(self, batch_size = 1000):
        if not hasattr(self, 'file_obj'):
            self.open_next()
        # a batch_group is a dictionary with keys 'key_id' and 'embeddings'
        # we will use self.batch_idx to keep track of the current index in the batch_group
        # starting from batch_idx, we will get a batch of size batch_size
        # if we reach the end of the batch_group, we will get the next batch_group
        # if we reach the end of the file, we will close the file and open the next one
        # if we reach the end of the files, we will return a StopIteration exception
        batch_group = self.batch_group
        batch_idx = self.batch_idx
        batch_size = min(batch_size, len(batch_group['key_id']) - batch_idx)
        key_ids = batch_group['key_id'][batch_idx:batch_idx+batch_size]
        embeddings = batch_group['embeddings'][batch_idx:batch_idx+batch_size]
        self.batch_idx += batch_size
        if self.batch_idx == len(batch_group['key_id']):
            try:
                self.batch_key = next(self.batch_iterator)
                self.batch_group = self.file_obj[self.batch_key]
                self.batch_idx = 0
            except StopIteration:
                self.file_obj.close()
                try:
                    self.open_next()
                except StopIteration:
                    raise StopIteration
        return key_ids, embeddings
    
    def __get__(self, batch_size = 1000):
        return self.get_batch(batch_size)
    

def get_merged_df(path_pattern= '../data/_power_outlet/*.csv'):
    csv_files = glob(path_pattern)
    filtered_dfs = []
    
    for i,file in enumerate(csv_files):
        print(f'{i*100/len(csv_files):.1f}%', file, end='\r')
        # Read the CSV file
        df = pd.read_csv(file)
    
        # Add the filtered dataframe to the list
        filtered_dfs.append(df[['key_id', 'countrycode', 'locale', 'duration']])
    
    # Concatenate all filtered dataframes
    merged_df_info = pd.concat(filtered_dfs, ignore_index=True)
    return merged_df_info

def get_uniform_df(df, column, num_samples=1000, column_values=None, top_column_values=None, allow_less_samples=True):
    """
    Construct a dataframe by sampling uniformly from `column` in `df`.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame to sample from.
    - column (str): The name of the column whose distribution needs to become uniform (e.g., 'countrycode').
    - num_samples (int): The number of samples to keep from each category in `column`.
    - column_values (list): Specify if only a specific subset of values in `column` should be used in sampling (e.g., specific list of countries).
    - top_column_values (int): Consider only the top N most frequent values in the column for sampling.
    - allow_less_samples (bool): Whether to allow some groups have less than `num_samples` samples (when the group has a small number of entries). 
        If `False`, only groups with More than `num_samples` datapoints will be sampled. 

    Returns:
    - pd.DataFrame: A uniformly sampled DataFrame.
    """
    
    # Filter the dataframe based on column_values if provided
    if column_values is not None:
        df = df[df[column].isin(column_values)]

    # Group the dataframe by the specified column
    grouped = df.groupby(column)

    # If top_column_values is specified, filter groups to keep only the top N most frequent
    if top_column_values is not None:
        top_values = df[column].value_counts().nlargest(top_column_values).index
        grouped = df[df[column].isin(top_values)].groupby(column)

    if allow_less_samples:
        filtered_groups = grouped
    else:
        # Filter groups by size (having at least num_samples)
        filtered_groups = grouped.filter(lambda x: len(x) >= num_samples).groupby(column)

    # Determine the minimum size among the qualified groups
    # min_size = min(filtered_groups.size().min(), num_samples)

    # Sample uniformly from each group
    uniform_samples = pd.DataFrame()
    for name, group in filtered_groups:
        # print(group.size)
        min_size = min(group.shape[0], num_samples)
        # print(min_size, group.shape[0])
        sample = group.sample(n=min_size, replace=False)
        uniform_samples = pd.concat([uniform_samples, sample], ignore_index=True)

    return uniform_samples


import pickle

# we want a modified Umap class which can embed new data points using the trained Umap
# this class will use the PCA class to project new data points to the PCA space of the training data
# it will also have a method to embed new data points using the trained Umap
# this one won't have the "training" and "rest" split of the previous class 
# and instead uses all of the training data to train Umap
class UmapEmbeddings:
    def __init__(self, emb_file, sampled_df, all_data_df, num_pc = 40):
        self.sampled_df = sampled_df
        self.all_data_df = all_data_df
        self.emb_file = emb_file
        self.name = self.get_name(emb_file)
        print(self.name)
        self.load_embeddings()
        print('doing PCA')
        self.do_PCA(num_pc)
        self.embed_loader = EmbeddingLoader(emb_file)
        
    def get_name(self,file_name):
        if type(file_name)==str:
            return file_name.split('/')[-1].split('-')[0].rstrip("_0123456789")
        elif type(file_name)==list:
            return file_name[0].split('/')[-1].split('-')[0].rstrip("_0123456789")
        
    def do_PCA(self, num_pc = 40):
        # we use the PCA class to do PCA
        # this allows us to project future data as well
        self.pca = PCA(self.embeddings, k=num_pc)
        self.pca_scores = self.pca.project(self.embeddings)
        
    def load_embeddings(self):
        print('Loading embeddings:\n=================')
        self.sample_key_ids_ = set(self.sampled_df.key_id.values) 
        self.emb = load_embedding_h5_sampled(self.emb_file, sample_key_ids=self.sample_key_ids_)
        self.embeddings = self.emb['embeddings']
        self.key_ids = np.array(self.emb['key_id'])
        self.key_ids_set = set(self.key_ids)
        print('Shape of embedding points', self.embeddings.shape)
        assert len(self.key_ids) == len(self.embeddings)
    
    def train_umap(self, epochs = 1000, dims = 2):
        # it's better to take the fraction for embedding as an input here
        self.umap_reducer = umap.UMAP(n_neighbors=15, n_components=dims, learning_rate=3.,n_epochs=epochs, verbose=1)
        self.umap_results = self.umap_reducer.fit_transform(self.pca_scores)
        self.umap_key_ids = self.key_ids
        
    # we want a method to embed new data points using the trained Umap
    def embed_new_data(self, new_data):
        # convert new_data to torch tensor
        new_data = torch.tensor(new_data)
        # project new data to PCA space
        new_data_pca = V(self.pca.project(new_data))
        # embed new data using the trained Umap
        new_data_umap = self.umap_reducer.transform(new_data_pca)
        return new_data_umap
    
    # embed new data points using the trained Umap
    def embed_data_from_loader(self, batch_size=1000, total_size=None):
        """starts loading data from the loader and embedding it using the trained Umap
        until the total_size is reached or the loader is exhausted
        
        returns: dictionary with key_ids as keys and umap embeddings as values
        """
        new_data_umap = []
        new_data_key_ids = []
        # if total_size is not provided, we will loop until the loader is exhausted
        if total_size is None:
            total_size = 1e12
        current_size = 0
        while current_size < total_size:
            try:
                key_ids, embeddings = self.embed_loader.get_batch(batch_size)
            except StopIteration:
                break
            # upon KeyboardInterrupt, we want to save the current state
            # we want this to return the key_ids as well
            # try:
            new_data_umap.append(self.embed_new_data(embeddings))
            new_data_key_ids.append(key_ids)
            current_size += len(key_ids)
            print(f'Embedded {current_size} data points', end='\n')
            # except KeyboardInterrupt:
            #     break
        new_data_umap = np.vstack(new_data_umap)
        new_data_key_ids = np.concatenate(new_data_key_ids)
        return {'key_ids': new_data_key_ids, 'umap': new_data_umap}
    
    # a method to embed all data points using the trained Umap and save the results as a merged_df
    def embed_all_data(self, save_dir = 'umap_embeddings/', batch_size=10_000, save_every=100_000, 
                umap_epochs = 500, umap_dims = 2):
        # first, assert that the umap has been trained
        if not (hasattr(self, 'umap_reducer') and hasattr(self, 'umap_results')):
            # we first need to run the training step
            self.train_umap(epochs=umap_epochs, dims=umap_dims)
            
        # make a dir using the save_dir and self.name if save_name is not provided
        save_dir = os.path.join(save_dir,  self.name + '_umap_embeddings_dim_' + str(umap_dims))
        os.makedirs(save_dir, exist_ok=True)
        self.save_train(save_dir)
        
        # we will loop over the loader and embed the data points
        # we will save the results every save_every data points
        # we will also save the results at the end
        current_size = 0
        while True:
            try:
                new_data = self.embed_data_from_loader(batch_size=batch_size, total_size=save_every)
                current_size += len(new_data['key_ids'])
                print(f'Embedded {current_size} data points', end='\n')
                file_name = os.path.join(save_dir, f'{current_size}_umap_embeddings.csv')
                self.save_new_data(new_data, file_name)
            except StopIteration:
                break
            
            file_name = os.path.join(save_dir, f'{current_size}_umap_embeddings.csv')
            self.save_new_data(new_data, file_name)
        
        # save the final results
        file_name = os.path.join(save_dir, f'{current_size}_umap_embeddings.csv')
        self.save_new_data(new_data, file_name)
        
    def save_new_data(self, new_data, file_name):
        new_data_df = self.get_merged_df(new_data['umap'], new_data['key_ids'])
        new_data_df.to_csv(file_name, index=False)
        
    def get_merged_df(self, umap_results, umap_key_ids):
        # Assuming clust.umap_key_ids is a list of key_ids corresponding to each point in umap_results
        dim = umap_results.shape[-1]
        umap_df = pd.DataFrame(umap_results, columns=[f'UMAP_{i}' for i in range(1,dim+1)])
        umap_df['key_id'] = umap_key_ids
        
        # Merge with merged_df to get countrycodes
        # merged_df = pd.merge( self.sampled_df[['key_id', 'countrycode','locale']], umap_df, on='key_id')
        merged_df = pd.merge( self.all_data_df[['key_id', 'countrycode','locale']], umap_df, on='key_id')
        return merged_df
    
    # we also want a method to save the trained Umap and PCA
    def save_train(self, save_dir):
        self.merged_df = self.get_merged_df(self.umap_results, self.umap_key_ids)
        file_name_base = os.path.join(save_dir, self.name)
        print('saving to file:\n \t', file_name_base )
        self.merged_df.to_csv(file_name_base + '_umap_train.csv', index=False)

        # save the trained Umap
        with open(file_name_base + '_umap_reducer.pkl', 'wb') as f:
            pickle.dump(self.umap_reducer, f)
        # save the trained PCA
        with open(file_name_base + '_pca.pkl', 'wb') as f:
            pickle.dump(self.pca, f)    

    def _plot_range(self,p=2.7):
        s = self.umap_results.std(0)
        m = self.umap_results.mean(0)
        lo,hi = m-p*s,m+p*s
        plt.xlim(lo[0],hi[0])
        plt.ylim(lo[1],hi[1])

    def plot_umap(self,range_factor = 2.7, skip=1, markersize=5, fontsize=8, alpha=.6): 
        dat = self.umap_results[::skip]
        plt.scatter(*dat.T, s=markersize, alpha=alpha,) 
        self._plot_range(range_factor)
        
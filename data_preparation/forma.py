import pandas as pd
# import timm
import time
import numpy as np
import json
from glob import glob
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
# import pytorch_lightning as pl
# from numpy import *

import h5py


V = lambda x: x.detach().cpu().numpy()

class Time:
    def __init__(self):
        self.set_t0()
        
    def set_t0(self):
        self.t0 = time.time()
        
    def dt(self):
        dt = time.time()-self.t0
        # print(f'{dt:.5g} s')
        return dt
    
T = Time()

#===== functions =====

class DrawDatasetAllFiles(Dataset):
    def __init__(self,feats_files_pattern, data_config, 
                 chunksize = 10000,
                 row_start = 0, 
                 filler_val = -0.5,
                 pix_max = 0.5,
                 res=128, interp_factor=8):
        
        """Dataset for reading multiple csv files by chunks and convert them to drawings. 
        feats_files_pattern: regex pattern of files to use (e.g. './data/*.csv')
        data_config: config of timm model used for embedding to set image dimensions etc.
        chunksize: (10000) size of chunk of csv file to read each time. 
        filler_val: (-0.5) what value to put in empty pixels. 
        pix_max: (0.5) max pixel value, before normalization. 
        row_start: (0) set if some rows at the start of csv need to be skipped. 
        res: (128) resolution of drawings created from data. 
            (further scaled to data_config resolution (e.g. to 512px for Dino))  
        interp_factor: (8) interpolation factor, used to fill gaps between data points  
            in the drawing data. 
        """
        
        self.row_start = row_start
        self.chunksize = chunksize
        self.prep_files(feats_files_pattern)
        
        # self.df = self.get_chunk()
        # self.load_chunk()
        
        self.res = res 
        self.interp_factor = interp_factor
    
        # self.device = device 
        self.data_config = data_config
        self.ch_mean = torch.tensor(data_config['mean']).view(1,-1, 1, 1)#.to(device) 
        self.ch_std = torch.tensor(data_config['std']).view(1,-1, 1, 1)#.to(device) 

        self.filler_val = filler_val 
        self.pix_max = pix_max 

    def prep_files(self, feats_files_pattern):
        print("Getting file list from ", feats_files_pattern) 
        self.files = glob(feats_files_pattern)
        print(f"{len(self.files)} files to process.")
        self._file_counter = 0
        self.files_iterator = iter(self.files)
        self.get_next_file()

    def get_next_file(self):
        self._feat_file = next(self.files_iterator) 
        self._df_loader = pd.read_csv(self._feat_file, skiprows = self.row_start, 
            # nrows=number_rows, 
            # usecols=["countrycode", "drawing"], 
            chunksize=self.chunksize)
        self.get_file_name_info()
        self._file_counter += 1
        print(f'Loaded file {self._file_counter}/{len(self.files)} \n\t', self._feat_file)
        
    def get_file_name_info(self):
        self.feat_name_numbered = os.path.split(self._feat_file)[-1].rstrip('.csv')#.rstrip('0123456789').rstrip('_')

        # All embeddings for one category are in a single h5 file. 
        # To do so, the "group" names should be modified from batch_0, batch_1, ... 
        # to something like batch_000000000012_0, .... using the file number 
        
        self.file_number = self.feat_name_numbered.split('_')[-1]
        
        # then the file_number should be stripped from the feat_name as:
        self.feat_name = self.feat_name_numbered.rstrip('0123456789').rstrip('_')
    
    def load_chunk(self):
        try: 
            df = self._df_loader.get_chunk()
        except StopIteration:
            try:
                self.get_next_file()
                df = self._df_loader.get_chunk()
            except StopIteration:
                print("Finished processing all files. \n" 
                      "==============================")
                return False
        self.df = df[df["drawing"]!="[]"] # remove empty drawings
        self.data = self.df.drawing.values
        self.ids = self.df.key_id.values
        print('Loaded new chunk of size ', len(self.data))
        return True
    
    def get_chunk(self,):
        df = self._df_loader.get_chunk()
        df = df[df["drawing"]!="[]"] # remove empty drawings
        return df
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        x = self.get_image_from_df(x, res=self.res, interp_factor=self.interp_factor)
        return self.pic2tensor(x[np.newaxis]), self.ids[idx]
        
    def fill_points(self, seg, factor=3,):
        filled = np.array([np.interp(np.linspace(0,1,factor*len(j)), np.linspace(0,1,len(j)), j) for j in seg])
        return torch.tensor(filled, dtype = torch.float32)

    def get_drawing_points(self, x, interp_factor=3,):
        # interpolate points on each segment
        x_filled = [self.fill_points(seg, factor=interp_factor) for seg in x]
        # concat segments
        return torch.cat(x_filled, dim=1)
    
    def get_image_from_points(self, points, res=64,):
        # get range
        xmax, xmin = points.max(1, keepdim=False)[0], points.min(1, keepdim=False)[0]
        dx = xmax - xmin

        max_range = dx[:2].max()

        # build image
        im = torch.ones((res, res), dtype = torch.float32) * self.filler_val

        # convert x,y to pixel values 
        offset = 0
        shifted_xy = (points - (1 - offset) * xmin[:, np.newaxis])[:2]
        rescaled_range = (1 + 2 * offset) * max_range
        idx = ((res - 1) * shifted_xy / (rescaled_range + 1e-0)).to(torch.long)
        # return idx
        pix_val = self.pix_max * (points[2] - xmin[2]) / (dx[2] + 1)
        im[idx[0], idx[1]] = pix_val

        return im

    def get_image_from_df(self, drawing_str, res = 128, interp_factor=10):
        x = json.loads(drawing_str)
        drawing_pts = self.get_drawing_points(x, interp_factor = interp_factor)
        return self.get_image_from_points(drawing_pts, res = res,)
        
    def get_all_im_tensors(self, df):
        ims = [self.get_image_from_df(i) for i in df.drawing]
        ims = torch.stack(ims)
        return ims
    
    def all_pics2tensor(self, pic_tensor, ):
        c,h,w = self.data_config['input_size'] 

        # Reshape
        pic_tensor = pic_tensor.unsqueeze(1).repeat(1, c, 1, 1)

        # Resize
        pic_tensor = F.interpolate(pic_tensor, size=(h,w), mode='bicubic', align_corners=False)
        
        # Normalize
        pic_mean = pic_tensor.mean(dim=(2,3),keepdim=True)
        pic_std = pic_tensor.std(dim=(2,3),keepdim=True)
        pic_tensor = (pic_tensor - pic_mean ) / pic_std * self.ch_std + self.ch_mean
        return pic_tensor
    
    def pic2tensor(self, pic_tensor, ):
        c,h,w = self.data_config['input_size'] 

        # Reshape
        # pic_tensor = pic_tensor.unsqueeze(0).repeat(1, c, 1, 1)
        pic_tensor = pic_tensor[np.newaxis].repeat(1, c, 1, 1)

        # Resize
        pic_tensor = F.interpolate(pic_tensor, size=(h,w), mode='bicubic', align_corners=False)
        
        # Normalize
        pic_mean = pic_tensor.mean(dim=(2,3),keepdim=True)
        pic_std = pic_tensor.std(dim=(2,3),keepdim=True)
        pic_tensor = (pic_tensor - pic_mean) / pic_std * self.ch_std + self.ch_mean
        return pic_tensor[0]

    def get_data(self,):
        df = self.get_chunk()
        return self.get_all_im_tensors(df)
        
#!!! Need to cange save_h5. Prior to running a batch, 
# we want to know if that batch has already been saved.
# This can be done by `try: f.create_group` before running,  

def save_h5(fname, label, ids_list, embeddings):
    print('saving to :', fname)
    with h5py.File(fname, 'a') as f:
            group = f.create_group(label)
            group.create_dataset('key_id', data=np.array(ids_list))
            group.create_dataset('embeddings', data=np.vstack(embeddings))
            
            
def load_embedding_h5(emb_file):
    with h5py.File(emb_file, 'r') as f:
        embeddings = []
        ids = []

        for batch_name in f.keys():
            batch_group = f[batch_name]
            batch_ids = batch_group['key_id'][:]
            batch_embeddings = batch_group['embeddings'][:]

            embeddings.append(torch.tensor(batch_embeddings))
            ids.extend(batch_ids)

        embeddings_tensor = torch.vstack(embeddings)
    
    return {'key_id': ids, 'embeddings': embeddings_tensor}

import pandas as pd
import numpy as np
import torch                    
from torch.utils.data import Dataset
import os
from PIL import ImageFile,Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

class MemesDataset(Dataset):
    #used to create imaging tensors for the Hateful memes data

    def __init__(self, data_frame, root_dir, transform=None):
        self.data_frame = data_frame
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 1])
        image = Image.open(img_name)
        image = image.convert('RGB')

        label = self.data_frame.iloc[idx, -2]
        
        if self.transform:
            image = self.transform(image)
    
        return (image, label)
    
class AmazonImgDataset(Dataset):
    #used to create imaging dataset for amazon reviews 

    def __init__(self, data_frame, transform=None):
        self.data_frame = data_frame
        self.transform = transform
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.data_frame.iloc[idx, -2] 
            
        image = Image.open(img_name)
        image = image.convert('RGB')

        label = self.data_frame.iloc[idx, -4]
        
        if self.transform:
            image = self.transform(image)
    
        return (image, label)
    
class AmazonTabDataset(Dataset):
    #used to create tabular dataset for amazon metadata
    
    def __init__(self, df):
        
        self.categorical = ["year", "main_cat", "brand", "verified"]
        self.target = "sentiment"
     
        self.frame = pd.get_dummies(df, columns=self.categorical)

        self.X = self.frame.drop([self.target,"reviewerID", "asin"], axis=1)
        self.y = self.frame[self.target]

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
       
        return [self.X.iloc[idx].values, self.y.iloc[idx]]   



class TCGA_TabDataset(Dataset):
    #used to create tabular datasets for the TCGA data

    def __init__(self, m, s, e, COMBINED_DATA_PATH):
        """
        Inputs: 
            m -> modality (e.g., "clinical", "images", etc.)
            s -> split (e.g., "test", "val", "train" )
            e -> random forest estimator (e.g., 50, 100, 150, etc.)

        """

        # solutions
        splits = pd.read_csv(COMBINED_DATA_PATH +  "splits.csv")
        splits = splits[splits["split"] == s]
        splits.y = splits.y.map(dict(lung =1, kidney =3, liver =2, stomach=1, colon=0)).astype(int) 
        #data
        if m != "clinical":
            self.data = pd.read_csv(COMBINED_DATA_PATH + "split_data/reduced/" + m + "_" + str(e) + "_"  + s + ".csv") 
        else:
            self.data = pd.read_csv(COMBINED_DATA_PATH + "split_data/reduced/" + m + "_"  + s + ".csv") 
            self.data = self.data.drop(columns=["y"])
        
        # drop solutions and case ids off data
        case = "case_id"
        self.data = self.data.drop(columns=[case])
        print(len(self.data.columns))
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        return [self.data.iloc[idx].values, self.sols[idx]]        
    
class TCGA_ImgDataset(Dataset):
    #used to create tabular datasets for the TCGA data

    def __init__(self, data_frame, transform=None):
        self.data_frame = data_frame
        self.transform = transform
    
    def __len__(self):
        # Return the length of the dataset
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        # Return the observation based on an index. Ex. dataset[0] will return the first element from the dataset, in this case the image and the label.
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = self.data_frame.iloc[idx, 3]
        image = Image.open(img_name)
        image = image.convert('RGB')
        image = np.array(image)

        label = self.data_frame.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)
    
        return (image, label)
    
"""
*************************************************************************************************************************************************************
The following functions were taken from: https://github.com/huanghoujing/pytorch-wrapping-multi-dataloaders/blob/master/wrapping_multi_dataloaders.py
"""
class MyIter(object):
    """An iterator."""
    def __init__(self, my_loader):
        self.my_loader = my_loader
        self.loader_iters = [iter(loader) for loader in self.my_loader.loaders]

    def __iter__(self):
        return self

    def __next__(self):
        # When the shortest loader (the one with minimum number of batches)
        # terminates, this iterator will terminates.
        # The `StopIteration` raised inside that shortest loader's `__next__`
        # method will in turn gets out of this `__next__` method.
        batches = [loader_iter.next() for loader_iter in self.loader_iters]
        return self.my_loader.combine_batch(batches)

        # Python 2 compatibility
        next = __next__

    def __len__(self):
        return len(self.my_loader)


class MyLoader(object):
    """This class wraps several pytorch DataLoader objects, allowing each time 
    taking a batch from each of them and then combining these several batches 
    into one. This class mimics the `for batch in loader:` interface of 
    pytorch `DataLoader`.
    Args: 
    loaders: a list or tuple of pytorch DataLoader objects
    """
    def __init__(self, loaders):
        self.loaders = loaders

    def __iter__(self):
        return MyIter(self)

    def __len__(self):
        return min([len(loader) for loader in self.loaders])

    # Customize the behavior of combining batches here.
    def combine_batch(self, batches):
        return batches


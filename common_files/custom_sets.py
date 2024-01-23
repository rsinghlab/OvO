import pandas as pd
import numpy as np
import torch                    
from torch.utils.data import Dataset
import os
from PIL import ImageFile,Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

class tadpoleDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, labels, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): A Pandas DataFrame containing the data.
            labels (pd.DataFrame): A Pandas DataFrame containing the labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataframe = dataframe
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        '''
        Returns a tuple (sample, label)
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = torch.tensor(self.dataframe.iloc[idx], dtype=torch.float)
        label = torch.tensor(self.labels.iloc[idx], dtype=torch.float)

        if self.transform:
            sample = self.transform(sample)

        return sample, label
    
class eICUDataset(Dataset):
    def __init__(self, file_path):
        #target_mapping={"Expired": 0, "Alive": 1}
        # Load the data
        df = pd.read_csv(file_path)

        # Map target variable to int
        #df['target'] = df['target'].map(target_mapping)
        self.labels = df['target']
        
        columns_to_drop = ['Unnamed: 0', 'patientunitstayid', 'uniquepid', 'target']
        columns_to_drop = [col for col in columns_to_drop if col in df.columns]

        # Drop the specified columns
        self.data = df.drop(columns=columns_to_drop, errors='ignore')

        # Convert to tensors
        self.data = torch.tensor(self.data.values, dtype=torch.float32)
        self.labels = torch.tensor(self.labels.values, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def save(self, file_path):
        torch.save(self, file_path)

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
        batches = [next(loader_iter) for loader_iter in self.loader_iters]
        return self.my_loader.combine_batch(batches)


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



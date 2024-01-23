import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import json
import os
import pickle

def load_dataset(file_path):
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, target_cols, max_token_len=512):
        self.labels = dataframe[target_cols].values
        self.tokenizer = tokenizer
        self.dataframe = dataframe
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data_row = self.dataframe.iloc[idx]

        text_data = data_row['text']
        
        encoding = self.tokenizer.encode_plus(
            text_data,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        text = encoding['input_ids'].flatten()
        att_mask = encoding['attention_mask'].flatten()
        labels = torch.tensor(self.labels[idx], dtype=torch.float)
        return text, att_mask, labels

class TimeSeriesDataset(Dataset):
    def __init__(self, time_series_data, labels):
        self.time_series_data = time_series_data
        self.labels = labels

    def __len__(self):
        return len(self.time_series_data)

    def __getitem__(self, idx):
        time_series = self.time_series_data[idx]
        label = self.labels[idx]
        ts = torch.tensor(time_series, dtype=torch.float)
        labels = torch.tensor(label, dtype=torch.float)
        return ts, labels


class DemographicsDataset(Dataset):
    def __init__(self, dataframe, target_cols):
        self.labels = dataframe[target_cols].values
        self.features = dataframe.drop(columns=target_cols).values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        demo = torch.tensor(self.features[idx], dtype=torch.float)
        labels = torch.tensor(self.labels[idx], dtype=torch.float)
        return demo, labels


class MedicalImageDataset(Dataset):
    def __init__(self, dataframe, img_col, target_cols, transform=None):
        self.dataframe = dataframe
        self.img_col = img_col
        self.target_cols = target_cols
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx][self.img_col]
        image = Image.open(img_path).convert('RGB')  # Convert to RGB for consistency

        if self.transform:
            image = self.transform(image)

        labels = self.dataframe.iloc[idx][self.target_cols]

        # Ensure labels are numeric and handle any non-numeric cases
        labels = labels.apply(pd.to_numeric, errors='coerce').fillna(0).values
        labels = torch.tensor(labels, dtype=torch.float)

        return image, labels

def regular_sample_sequence(seq, fixed_length):
    n = len(seq)
    if n > fixed_length:
        # Select indices at regular intervals
        indices = np.round(np.linspace(0, n - 1, fixed_length)).astype(int)
        sampled_seq = seq[indices]
    else:
        # If shorter, pad the sequence
        padded_seq = np.zeros((fixed_length, seq.shape[1]))
        padded_seq[:n] = seq
        sampled_seq = padded_seq
    return sampled_seq



def preprocess_timeseries(file_path, config, encoders): #, fixed_length
    categorical_cols = [col for col in config["id_to_channel"] if config["is_categorical_channel"][col]]
    continuous_cols = [col for col in config["id_to_channel"] if not config["is_categorical_channel"][col]]

    ts = pd.read_csv(file_path)

    # Impute missing values
    ts[continuous_cols] = ts[continuous_cols].fillna(method='ffill').fillna(0)
    ts[categorical_cols] = ts[categorical_cols].fillna('missing')
    
    # Initialize a DataFrame for the encoded data
    encoded_data = pd.DataFrame(index=ts.index)

    # One-hot encode each categorical column separately
    for col in categorical_cols:
        if col in encoders:  # Check if the encoder for the column exists
            encoded_col = encoders[col].transform(ts[[col]])
            encoded_col_df = pd.DataFrame(encoded_col, columns=encoders[col].get_feature_names_out([col]), index=ts.index)
            encoded_data = pd.concat([encoded_data, encoded_col_df], axis=1)

    ts = ts.drop(columns=categorical_cols)
    ts = pd.concat([ts, encoded_data], axis=1)
    
    # Normalize continuous variables
    scaler = StandardScaler()
    ts[continuous_cols] = scaler.fit_transform(ts[continuous_cols])
    
    # Regular sample or pad sequence
    #ts = [regular_sample_sequence(data.values, fixed_length) for data in ts] #regular_sample_sequence(ts, fixed_length)
    #print(ts[0].shape)
    #print(ts[20].shape)
    return ts


def pad_sequence(seq, maxlen, n_features):
    padded_seq = np.zeros((maxlen, n_features))
    padded_seq[:len(seq)] = seq[:maxlen]
    return padded_seq


def preprocess_demo(df, target_cols, categorical_encoders=None):
    # Identify categorical and continuous columns
    demographic_cols = ['admittime', 'dischtime', 
       'admission_type', 'admission_location', 
       'insurance', 'language', 'marital_status', 'ethnicity', 'gender', 'anchor_age',
       'anchor_year', 'anchor_year_group', 'deathtime','discharge_location']
    df = df[target_cols + demographic_cols]  # Demographic information

    continuous_cols = df[demographic_cols].select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df[demographic_cols].select_dtypes(include=['object']).columns

    # Handle missing values
    imputer_continuous = SimpleImputer(strategy='mean')
    imputer_categorical = SimpleImputer(strategy='most_frequent')

    df[continuous_cols] = imputer_continuous.fit_transform(df[continuous_cols])
    df[categorical_cols] = imputer_categorical.fit_transform(df[categorical_cols])

    # One-hot encode categorical variables
    if categorical_encoders is None:
        categorical_encoders = {}
        for col in categorical_cols:
            encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
            categorical_encoders[col] = encoder.fit(df[[col]])
            transformed = encoder.transform(df[[col]])
            df = df.drop(columns=[col])
            df = pd.concat([df, pd.DataFrame(transformed, columns=encoder.get_feature_names_out([col]))], axis=1)
    else:
        for col in categorical_cols:
            encoder = categorical_encoders[col]
            transformed = encoder.transform(df[[col]])
            df = df.drop(columns=[col])
            df = pd.concat([df, pd.DataFrame(transformed, columns=encoder.get_feature_names_out([col]))], axis=1)


    # Normalize continuous variables
    scaler = StandardScaler()
    df[continuous_cols] = scaler.fit_transform(df[continuous_cols])
    object_cols = df.select_dtypes(include=['object']).columns
    print(object_cols)
    
    return df, categorical_encoders




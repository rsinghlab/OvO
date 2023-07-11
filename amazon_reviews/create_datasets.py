import sys
sys.path.append('../..')
import pandas as pd
import torch
import glob                    
from torchvision import transforms
from torch.utils.data import TensorDataset
from common_files.model_utils import tokenize_mask
from common_files.custom_sets import AmazonImgDataset,AmazonTabDataset


def create_img_embeddings(split, df, path):
    """
    Creates image PyTorch datasets and save them. 

    Parameters:
    split (str): The split (e.g. "train", "dev", "test") to create embeddings for.
    df (pandas.DataFrame): The data frame containing the metadata for the reviews.
    """
    
    img_inputs = AmazonImgDataset(
        data_frame=df,
        transform=transforms.Compose([
            transforms.Resize((100,100)),
            transforms.ToTensor(),
        ])
    )
    save_to = path + split + "_img_inputs.pt"
    torch.save(img_inputs, save_to)
    
    

def clean_df_tabular(df):
    """
    Cleans a tabular data frame by imputing missing values and reducing the number of categories in some columns.

    Parameters:
    df (pandas.DataFrame): The data frame to be cleaned.

    Returns:
    pandas.DataFrame: The cleaned data frame.
    """
   
    df.loc[df['main_cat'].isin((df['main_cat'].value_counts()[df['main_cat'].value_counts() < 15]).index),
       'main_cat'] = 'Other'
    df["main_cat"] = df["main_cat"].fillna("Other")
    
    df.loc[df['brand'].isin((df['brand'].value_counts()[df['brand'].value_counts() < 155]).index),
       'brand'] = 'Other'
    df["brand"] = df["brand"].fillna("Other")
    
    df["year"] = df["date"].str[-4:]
    
    df["vote"] = df["vote"].fillna('0', inplace = False)
    df["vote"] = df["vote"].astype(str).str.replace(",", "").astype(float).astype(int)
    
    df["price"] = df["price"].fillna("-1")
    df['price'] = df['price'].apply(lambda x: x if len(x) < 10 else "-1")
    df["price"] = df["price"].str.replace("$", "").str.replace(",", "").astype(float)
    temp = df[df["price"] != -1]
    brand_prices = temp.groupby("brand")["price"].mean().to_dict()
    df.loc[df['price'] == -1, 'price'] = df[df['price'] == -1]['brand'].map(brand_prices) #replace missing prices with the average price of each respective brand
    
    return df

def create_tabular_embeddings(split, df, path):
    """
    Creates a custom PyTorch dataset for tabular metadata and saves them.

    Parameters:
    split (str): The split (e.g. "train", "dev", "test") to create embeddings for.
    df (pandas.DataFrame): The data frame containing the cleaned metadata for the reviews.
    """

    df = df.drop(["title", "description", "overall",
         "reviewText", "category", "date", "feature", "reviewTime", "summary", "prd_img_path",
        "rvw_img_path"], axis=1)
    df = df[["reviewerID", "asin", "price", "year", "main_cat", "brand", "vote", "verified", "sentiment"]]
    
    dataset = AmazonTabDataset(df)
    
    save_to = path + split + "_tab_inputs.pt"
    torch.save(dataset, save_to)

def create_txt_embeddings(split, df, path):
    """
    Creates text embeddings and saves them to a PyTorch tensor.

    Parameters:
    split (str): The split (e.g. "train", "dev", "test") to create embeddings for.
    df (pandas.DataFrame): The data frame containing the reviews.
    """
    
    column = "reviewText"
    
    sentences = df[column].values
    input_ids, att, labels = tokenize_mask(sentences, df["sentiment"].values)
    df_inputs = TensorDataset(input_ids, att, labels)
    
    save_to = path + split + "_txt_inputs.pt"
    torch.save(df_inputs, save_to)

def main():
    """
    Command-line arguments:
    - path (str): the path to the main data directory
    """
    args = sys.argv[1:]
    path = args[0]

    df = pd.read_csv(path + "metadata_sentiment.csv")
    img_paths = glob.glob(path + "/rvw_images"  + "/**/*.jpg", recursive=True)
    df = df[df["rvw_img_path"].isin(img_paths)] #filtering for only images that were downloaded
    df = clean_df_tabular(df)
    df.to_csv(path + "sentiment_final.csv", index =False)

    df = df[df.sentiment.isin([0,2])] # we only end up using postive and negative sentiment, so we remove neutral
    df.sentiment = df.sentiment.replace({2:1}) 
    
    #80% for train, 10% for test and validation
    train = df.sample(frac = 0.8)
    test = df.drop(train.index).sample(frac = 0.5)
    dev = df.drop(train.index).drop(test.index)

    dic = {"train":train, "test":test, "dev":dev}
    for split in dic:
        df = dic[split]
        create_tabular_embeddings(split, df, path)
        create_txt_embeddings(split, df, path)
        create_img_embeddings(split, df, path)
        

if __name__ == "__main__":
    main()


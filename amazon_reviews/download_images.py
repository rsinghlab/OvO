import pandas as pd
from io import BytesIO
from PIL import Image
import requests
import numpy as np
import sys

def main():
    """
    This function downloads electronics reviews and metadata from Amazon and preprocesses the data for multi-modal sentiment analysis.
    Download the data from https://nijianmo.github.io/amazon/ by clicking on the "Electronics" section. Click on both the electornics reviews and electronics metadata.

    Command-line arguments:
    - path (str): the path to the main data directory
    """
    args = sys.argv[1:]
    data_path = args[0]

    meta = pd.read_json(data_path + 'meta_Electronics.json', lines=True)
    rvwr = pd.read_json(data_path + 'Electronics.json', lines=True)

    r = rvwr[~rvwr["image"].isna()]
    m = meta[meta.imageURLHighRes.str.len() != 0]
    df = m.merge(r, on = "asin")
    df = df[df["date"].str[-4:].isin(['2012','2013', '2014', '2015','2016', '2017', '2018'])]
    df["overall"] = df["overall"] -1
    df = df[df["description"].str.len() != 0]
    df = df[df["reviewText"].str.len() != 0]
    df = df.dropna(subset=["reviewText", "description"])

    df.to_csv(data_path + "full_electronic_2012-2018.csv", index=False)

    print(df["overall"].value_counts())

    conditions = [
        (df['overall'] < 2),
        (df['overall'] == 2),
        (df['overall'] > 2)
    ]

    values = [0, 1, 2] #0 -> negative, 1 -> neutral, 2 -> positive

    df['sentiment'] = np.select(conditions, values)

    df = df.groupby('sentiment').apply(lambda x: x.sample(n=11090)).reset_index(drop = True)
    df = df.sample(frac=1).reset_index(drop=True) #shuffle


    df.to_csv(data_path + "sampled_sentiment.csv", index = False)

    df["rvw_img_path"] = data_path + "rvw_images/" + df["reviewerID"] + "_" + df["asin"] + ".jpg"

    urls_df = df[["rvw_img_path", "image"]]

    urls_df["image"] = urls_df["image"].str[1:].str.split(",").apply(lambda x: x[0]).str.replace("'", '').str.replace("[", "").str.replace("]", "")

    urls_df = urls_df.drop_duplicates().reset_index().drop("index",axis=1)

    for i in range(len(urls_df)):
        image_url = urls_df["image"][i]
        image_path = urls_df["rvw_img_path"][i]
        img_data = requests.get(image_url)
        try:
            img_data = Image.open(BytesIO(img_data.content))
            img_data.save(image_path)
        except:
            print("file not found")
            

    df = df.drop(["tech1", "tech2", "also_buy", "rank", "fit", "similar_item", "style","reviewerName", "details", "unixReviewTime", "also_view", "imageURL", "imageURLHighRes", "image"],axis=1)

    df.to_csv(data_path + "metadata_sentiment.csv", index=False)

    mini = df[["asin", "reviewerID", "prd_img_path", "rvw_img_path", "description", "reviewText", "overall", "sentiment"]]
    mini.to_csv(data_path + "sentiment.csv", index=False)
    
if __name__ == "__main__":
    main()


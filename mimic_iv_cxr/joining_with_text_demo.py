import argparse
import pandas as pd
import numpy as np
import json
import glob
from data_utils import *
from sklearn.preprocessing import OneHotEncoder
from transformers import AutoTokenizer
import warnings
import pickle

"""
IMPORTANT NOTE: Before running this script, you must first run the preprocessing and test/train splitting that was done in MedFuse:
https://github.com/nyuad-cai/MedFuse/tree/main
Following their preprocessing will generate the needed files used here. Mainly, we use their listfiles and their general file path structure.
For the later stage preprocessing (e.g., tokenizing), we use our own functions in this file. 
Furthermore, this uses demographic and text notes from other MIMIC-IV datasets not mentioned in MedFuse. 
You will need the admissions and patients tables from MIMIC-IV, and the discharge notes: https://physionet.org/content/mimic-iv-note/2.2/note/#files-panel. 
"""

warnings.filterwarnings('ignore', category=UserWarning, message='.*OneHotEncoder was fitted without feature names.*')

CLASSES = [
       'Acute and unspecified renal failure', 'Acute cerebrovascular disease',
       'Acute myocardial infarction', 'Cardiac dysrhythmias',
       'Chronic kidney disease',
       'Chronic obstructive pulmonary disease and bronchiectasis',
       'Complications of surgical procedures or medical care',
       'Conduction disorders', 'Congestive heart failure; nonhypertensive',
       'Coronary atherosclerosis and other heart disease',
       'Diabetes mellitus with complications',
       'Diabetes mellitus without complication',
       'Disorders of lipid metabolism', 'Essential hypertension',
       'Fluid and electrolyte disorders', 'Gastrointestinal hemorrhage',
       'Hypertension with complications and secondary hypertension',
       'Other liver diseases', 'Other lower respiratory disease',
       'Other upper respiratory disease',
       'Pleurisy; pneumothorax; pulmonary collapse',
       'Pneumonia (except that caused by tuberculosis or sexually transmitted disease)',
       'Respiratory failure; insufficiency; arrest (adult)',
       'Septicemia (except in labor)', 'Shock'
    ]


def create_demo_general(path_to_core):
    pts = pd.read_csv(path_to_core + "patients.csv")
    admissions = pd.read_csv(path_to_core + "admissions.csv")
    demo = admissions.merge(pts, on = "subject_id")
    demo = demo[['subject_id', 'hadm_id', 'admittime', 'dischtime', 
       'admission_type', 'admission_location', 
       'insurance', 'language', 'marital_status', 'ethnicity', 'gender', 'anchor_age',
       'anchor_year', 'anchor_year_group','deathtime','discharge_location']]
    return demo

def create_groups(task, ehr_data_dir, cxr_data_dir):
    data_dir = cxr_data_dir
    cxr_metadata = pd.read_csv(f'{data_dir}/mimic-cxr-2.0.0-metadata.csv')
    icu_stay_metadata = pd.read_csv(f'{ehr_data_dir}/per_subject/all_stays.csv')
    columns = ['subject_id', 'stay_id', 'intime', 'outtime', 'hadm_id']
    # only common subjects with both icu stay and an xray
    cxr_merged_icustays = cxr_metadata.merge(icu_stay_metadata[columns ], how='inner', on='subject_id')

    # combine study date time
    cxr_merged_icustays['StudyTime'] = cxr_merged_icustays['StudyTime'].apply(lambda x: f'{int(float(x)):06}' )
    cxr_merged_icustays['StudyDateTime'] = pd.to_datetime(cxr_merged_icustays['StudyDate'].astype(str) + ' ' + cxr_merged_icustays['StudyTime'].astype(str) ,format="%Y%m%d %H%M%S")

    cxr_merged_icustays.intime=pd.to_datetime(cxr_merged_icustays.intime)
    cxr_merged_icustays.outtime=pd.to_datetime(cxr_merged_icustays.outtime)
    end_time = cxr_merged_icustays.outtime
    if task == 'in-hospital-mortality':
        end_time = cxr_merged_icustays.intime + pd.DateOffset(hours=48)

    cxr_merged_icustays_during = cxr_merged_icustays.loc[(cxr_merged_icustays.StudyDateTime>=cxr_merged_icustays.intime)&((cxr_merged_icustays.StudyDateTime<=end_time))]

    # select cxrs with the ViewPosition == 'AP
    cxr_merged_icustays_AP = cxr_merged_icustays_during[cxr_merged_icustays_during['ViewPosition'] == 'AP']

    groups = cxr_merged_icustays_AP.groupby('stay_id')

    groups_selected = []
    for group in groups:
        # select the latest cxr for the icu stay
        selected = group[1].sort_values('StudyDateTime').tail(1).reset_index()
        groups_selected.append(selected)
    groups = pd.concat(groups_selected, ignore_index=True)
    
    paths = glob.glob(cxr_data_dir + "/resized/" + '*.jpg')
    groups["dicom_id_path"] = cxr_data_dir + "/resized/" + groups["dicom_id"] + ".jpg"
    groups = groups[groups["dicom_id_path"].isin(paths)]
    return groups

def create_time_series_dataset(file_paths, config, encoders, fixed_length, labels):
    processed_data = [preprocess_timeseries(path, config, encoders) for path in file_paths]
    sampled_data = [regular_sample_sequence(data.values, fixed_length) for data in processed_data]
    print(sampled_data[0].shape)
    print(sampled_data[20].shape)
    
    return TimeSeriesDataset(sampled_data, labels)

def create_demo_dataset(pre_processed_data):
    return DemographicsDataset(pre_processed_data, CLASSES)

def create_image_dataset(data, img_col, transform):
    # Initialize the image dataset
    return MedicalImageDataset(data, img_col, CLASSES, transform)

def create_text_dataset(data):
    tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
    return TextDataset(data, tokenizer, CLASSES)

def determine_categories_and_length(ehr_data_dir, task, phase, data, config):
    categorical_cols = [col for col in config["id_to_channel"] if config["is_categorical_channel"][col]]

    all_categories = {col: set() for col in categorical_cols}
    lengths = []
    for file_path in data['stay']:
        full_path = f'{ehr_data_dir}/{task}/{phase}/' + file_path
        ts = pd.read_csv(full_path, usecols=categorical_cols)
        lengths.append(len(ts))
        for col in categorical_cols:
            all_categories[col].update(ts[col].dropna().unique())

    fixed_length = int(np.mean(lengths))

    return all_categories, fixed_length

def create_encoders(unique_categories):
    encoders = {}
    for col, categories in unique_categories.items():
        if not categories:
            print(f"No unique values found for column '{col}'. Skipping this column.")
            continue

        unique_values = list(categories)
        encoder = OneHotEncoder(categories=[unique_values], handle_unknown='ignore', sparse=False)
        encoder.fit(np.array(unique_values).reshape(-1, 1))

        encoders[col] = encoder

    return encoders


def load_and_preprocess_data(phase, groups, ds, demo, ehr_data_dir, cxr_data_dir, config, task, fixed_length=None, encoders=None, demo_train_data=None, demo_train_encoders=None):
    # Load phase-specific data and preprocess
    # Example: Load data files for 'train', 'val', or 'test'

    data_path = f'{ehr_data_dir}/{task}/{phase}_listfile.csv'
    data = pd.read_csv(data_path)
    data = data.merge(groups, on = ["stay_id"])
    data= data.merge(ds, on = ["subject_id", "hadm_id"])
    data= data.merge(demo, on = ["subject_id", "hadm_id"])
    labels = data[CLASSES].values

    if phase == 'train':
        # Determine unique categories and fixed_length for training data
        unique_categories, fixed_length = determine_categories_and_length(ehr_data_dir, task, phase, data, config)
        encoders = create_encoders(unique_categories)
        demo_train_data, demo_train_encoders = preprocess_demo(data, CLASSES, None)
        
    else:
        # Use provided fixed_length and encoders for validation and test data
        if fixed_length is None or encoders is None:
            raise ValueError("fixed_length and encoders must be provided for validation and test phases")
    
    return {
        'data': data, 
        'labels': labels,
        'fixed_length': fixed_length, 
        'encoders': encoders, 
        'demo_train_data':demo_train_data,
        'demo_train_encoders': demo_train_encoders
    }



def save_dataset(dataset, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(dataset, f)

def create_and_save_all_datasets(data, ehr_data_dir, task, phase, config):
    # Time series datasets
    if phase == "val":
        ts_file_paths = [f'{ehr_data_dir}/{task}/train/' + stay for stay in data['data']["stay"]]
    else:
        ts_file_paths = [f'{ehr_data_dir}/{task}/{phase}/' + stay for stay in data['data']["stay"]]
    labels = data["labels"]
    ts_dataset = create_time_series_dataset(ts_file_paths, config, data['encoders'], data['fixed_length'], labels)
    save_dataset(ts_dataset, f'{ehr_data_dir}/{task}/{phase}_ts_dataset.pkl')

    # Demo datasets
    if phase == "train":
        print(data['demo_train_data'].shape)
        demo_dataset = create_demo_dataset(data['demo_train_data'])
    else:
        demo_data, _ = preprocess_demo(data["data"], CLASSES, data['demo_train_encoders'])
        print(demo_data.shape)
        demo_dataset = create_demo_dataset(demo_data)

    save_dataset(demo_dataset, f'{ehr_data_dir}/{task}/{phase}_demo_dataset.pkl')

    # Image datasets
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_dataset = create_image_dataset(data['data'], 'dicom_id_path', transform)
    save_dataset(img_dataset, f'{ehr_data_dir}/{task}/{phase}_img_dataset.pkl')
    
    text_dataset = create_text_dataset(data['data'])
    save_dataset(text_dataset, f'{ehr_data_dir}/{task}/{phase}_text_dataset.pkl')


def main():
    parser = argparse.ArgumentParser(description="Process MIMIC datasets")
    parser.add_argument("--ehr_data_dir", type=str, required=True)
    parser.add_argument("--cxr_data_dir", type=str, required=True)
    parser.add_argument("--discharge_path", type=str, required=True)
    parser.add_argument("--core_dir", type=str, required=True)
    parser.add_argument("--config_file_path", type=str, required=True)
    parser.add_argument("--task", type=str, default="phenotyping")
    args = parser.parse_args()

    # Load configuration for preprocessing
    with open(args.config_file_path, 'r') as file:
        config = json.load(file)

    groups = create_groups(args.task, args.ehr_data_dir, args.cxr_data_dir)
    
    ds = pd.read_csv(args.discharge_path)
    
    demo = create_demo_general(args.core_dir)
    
    # Process train data and store fixed length and encoders
    train_data = load_and_preprocess_data('train', groups, ds, demo, args.ehr_data_dir, args.cxr_data_dir, config, args.task)
    create_and_save_all_datasets(train_data, args.ehr_data_dir, args.task, 'train', config)

    # Process validation and test data using fixed length and encoders from train data
    for phase in ['val', 'test']:
        data = load_and_preprocess_data(phase, groups, ds, demo, args.ehr_data_dir, args.cxr_data_dir, config, args.task, train_data['fixed_length'], train_data['encoders'], train_data['demo_train_data'], train_data['demo_train_encoders'])
        create_and_save_all_datasets(data, args.ehr_data_dir, args.task, phase, config)

if __name__ == "__main__":
    main()

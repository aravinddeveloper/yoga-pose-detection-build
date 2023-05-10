import argparse
import os,glob
import pandas as pd
from torch.utils import save_image
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from PIL import Image


BUCKET_NAME = "aravind-aws-ml-sagemaker"
PROCESSING_PATH = "/opt/ml/processing/"

def preprocess_image(args,list_of_file_and_label):
    df = pd.DataFrame(data=list_of_file_and_label,columns=["path","label"])
    x_train,x_test,y_train,y_test = train_test_split(df.path,df.label,test_size=args.train_test_split_ratio,random_state=42,shuffle=True,stratify=df.label)
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }  
    
    for image_path,label in zip(x_train,y_train):
        file_name = image_path.split("/")[-1]
        image_tensor = data_transforms["train"](Image.open(image_path))
        save_image(image_tensor,f"{PROCESSING_PATH}/{args.process_train_folder}/output/{label}/{file_name}")
        
    for image_path,label in zip(x_test,y_test):
        file_name = image_path.split("/")[-1]
        image_tensor = data_transforms["val"](Image.open(image_path))
        save_image(image_tensor,f"{PROCESSING_PATH}/{args.process_val_folder}/output/{label}/{file_name}")

def parse_args():
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--train-test-split_ratio",
        type=float,
        default=0.1,
        metavar="S",
        help="train test split ratio (default: 0.1)",
    )
    

    # Container environment
    parser.add_argument("--process-input-folder", type=str, default=PROCESSING_PATH+"input")
    parser.add_argument("--process-train-folder", type=str, default=PROCESSING_PATH+"train")
    parser.add_argument("--process-val-folder", type=str, default=PROCESSING_PATH+"val")
    
if __name__ == "__main__":
    args = parse_args()
    input_path_label_list = [(file,file.split("/")[-1].split("_")[0]) for file in glob.glob(args.process_input_folder+"/*.*") if os.path.isfile(file)]
    preprocess_image(args,input_path_label_list)
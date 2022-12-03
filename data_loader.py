import torch
import torch.nn as nn
import numpy as np
from nerf_utils.nerf import cumprod_exclusive, get_minibatches, get_ray_bundle, positional_encoding
from nerf_utils.tiny_nerf import VeryTinyNerfModel
from torchvision.datasets import mnist
from torchvision import transforms
import Lenet5
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from copy import deepcopy

### IMDB related

import pandas as pd
from sklearn.utils import shuffle
import re

def clean_text(text):
    text = re.sub(r"@[A-Za-z0-9]+", ' ', text)
    text = re.sub(r"https?://[A-Za-z0-9./]+", ' ', text)
    text = re.sub(r"[^a-zA-z.!?'0-9]", ' ', text)
    text = re.sub('\t', ' ',  text)
    text = re.sub(r" +", ' ', text)
    return text

def sentiment2label(sentiment):
    if sentiment == "positive":
        return 1
    else:
        return 0

from torch.utils.data import Dataset, DataLoader

class ImdbDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
        review,
        add_special_tokens=True,
        max_length=self.max_len,
        return_token_type_ids=False,
        pad_to_max_length=True,
        #pad_to_max_length=False,
        return_attention_mask=True,
        return_tensors='pt',
        )

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        """
        input_ids = pad_sequences(encoding['input_ids'], maxlen=self.max_len, dtype=torch.Tensor ,truncating="post",padding="post")
        input_ids = input_ids.astype(dtype = 'int64')
        input_ids = torch.tensor(input_ids)

        attention_mask = pad_sequences(encoding['attention_mask'], maxlen=self.max_len, dtype=torch.Tensor ,truncating="post",padding="post")
        attention_mask = attention_mask.astype(dtype = 'int64')
        attention_mask = torch.tensor(attention_mask)
        """

        return {
        'review_text': review,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'targets': torch.tensor(target, dtype=torch.long)
        }

def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = ImdbDataset(
    reviews=df.review.to_numpy(),
    targets=df.sentiment.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )

  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=4
  )

from transformers import XLNetTokenizer, XLNetModel
from transformers import XLNetForSequenceClassification
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
import transformers

def handle_imdb(device):
        df = pd.read_csv('/home/user/Documents/tau/wolf/datasets/IMDB Dataset.csv')
        df = shuffle(df)
        df['review'] = df['review'].apply(clean_text)
        df['sentiment'] = df['sentiment'].apply(sentiment2label)
        
        #tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        #model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=2)
        model_name = "prajjwal1/bert-tiny"
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)
        model = model.to(device)

        df_train, df_test = train_test_split(df, test_size=0.5, random_state=101)
        #df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=101)

        BATCH_SIZE = 1
        #MAX_LEN = 512
        MAX_LEN = 200
        train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
        #val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
        test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

        return train_data_loader, test_data_loader, model

def wrapper_dataset(config, args, device):
    if args.datatype == 'tinynerf':
        
        data =  np.load(args.data_train_path)
        images = data["images"]
        # Camera extrinsics (poses)
        tform_cam2world = data["poses"]
        tform_cam2world = torch.from_numpy(tform_cam2world).to(device)
        # Focal length (intrinsics)
        focal_length = data["focal"]
        focal_length = torch.from_numpy(focal_length).to(device)

        # Height and width of each image
        height, width = images.shape[1:3]

        # Near and far clipping thresholds for depth values.
        near_thresh = 2.0
        far_thresh = 6.0

        # Hold one image out (for test).
        testimg, testpose = images[101], tform_cam2world[101]
        testimg = torch.from_numpy(testimg).to(device)

        # Map images to device
        images = torch.from_numpy(images[:100, ..., :3]).to(device)
        num_encoding_functions = 10
        # Specify encoding function.
        encode = positional_encoding
        # Number of depth samples along each ray.
        depth_samples_per_ray = 32
        model = VeryTinyNerfModel(num_encoding_functions=num_encoding_functions)
        # Chunksize (Note: this isn't batchsize in the conventional sense. This only
        # specifies the number of rays to be queried in one go. Backprop still happens
        # only after all rays from the current "bundle" are queried and rendered).
        # Use chunksize of about 4096 to fit in ~1.4 GB of GPU memory (when using 8
        # samples per ray).
        chunksize = 4096
        batch = {}
        batch['height'] = height
        batch['width'] = width
        batch['focal_length'] = focal_length
        batch['testpose'] = testpose
        batch['near_thresh'] = near_thresh
        batch['far_thresh'] = far_thresh
        batch['depth_samples_per_ray'] = depth_samples_per_ray
        batch['encode'] = encode
        batch['get_minibatches'] =get_minibatches
        batch['chunksize'] =chunksize
        batch['num_encoding_functions'] = num_encoding_functions
        train_ds, test_ds = [],[]
        for img,tfrom in zip(images,tform_cam2world):
            batch['input'] = tfrom
            batch['output'] = img
            train_ds.append(deepcopy(batch))
        batch['input'] = testpose
        batch['output'] = testimg
        test_ds = [batch]
    elif args.datatype == 'mnist':
        model = Lenet5.NetOriginal()
        train_transform = transforms.Compose(
                            [
                            transforms.ToTensor()
                            ])
        train_dataset = mnist.MNIST(
                "\data\mnist", train=True, download=True, transform=ToTensor())
        test_dataset = mnist.MNIST(
                "\data\mnist", train=False, download=True, transform=ToTensor())
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1)
        train_ds, test_ds = [],[]
        for idx, data in enumerate(train_loader):
            train_x, train_label = data[0], data[1]
            train_x = train_x[:,0,:,:].unsqueeze(1)
            batch = {'input':train_x,'output':train_label}
            train_ds.append(deepcopy(batch))
        for idx, data in enumerate(test_loader):
            train_x, train_label = data[0], data[1]
            train_x = train_x[:,0,:,:].unsqueeze(1)
            batch = {'input':train_x,'output':train_label}
            test_ds.append(deepcopy(batch))
    elif args.datatype == 'imdb':
        train_ds, test_ds, model = handle_imdb(device)
    else:
        "implement on your own"
        pass
    return train_ds,test_ds,model

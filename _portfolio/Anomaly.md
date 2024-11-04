---
layout: post
title: Anomaly Detection in Industrial Machinery Using Sound Data
img: "assets/img/portfolio/Anomaly/S0.png"
date: 2023-10-03
---

{% include aligner.html images="portfolio/Anomaly/S0.png"%}
🏷️ PhD Research Project

### Listen: Can You Spot the Malfunction?
Let’s start with a quick test. Below are three example sound clips from industrial machinery. One of these sounds is from a malfunctioning machine. Listen carefully and see if you can tell which one:

- Sound clip 1:
<audio controls>
  <source src="{{ '/assets/img/portfolio/Anomaly/1. abnormal-False.wav' | relative_url }}" type="audio/wav">
  Your browser does not support the audio element.
</audio>

- Sound clip 2:
<audio controls>
  <source src="{{ '/assets/img/portfolio/Anomaly/2. abnormal-True.wav' | relative_url }}" type="audio/wav">
  Your browser does not support the audio element.
</audio>

- Sound clip 3:
<audio controls>
  <source src="{{ '/assets/img/portfolio/Anomaly/3. abnormal-False.wav' | relative_url }}" type="audio/wav">
  Your browser does not support the audio element.
</audio>

Think you know which one is malfunctioning? Keep this in mind as we explore how sound data can be used to detect anomalies in factory settings.

### Research Purpose
The goal of this small research was to reproduce the findings from a key study on anomaly detection in industrial machinery using sound data. Leveraging the MIMII Dataset, this research goal is to validate the effectiveness of using audio-based monitoring as a cost-effective and efficient method for detecting malfunctioning machinery in industrial environments.

The MIMII dataset is a collection of sound recordings from four types of industrial machines: valves, pumps, fans, and sliders. Each machine type is represented by datasets containing audio recordings of normal operations and four types of anomalies—contamination, leakage, rotating unbalance, and rail damage. 

<!-- {% include aligner.html images="portfolio/Anomaly/S1.png"%} -->
{% include aligner.html images="portfolio/Anomaly/S1.png" width="60%" height="auto" %}

For interested readers:
- *Original Paper Source: https://arxiv.org/pdf/1909.09347.pdf*
- *Original dataset link: https://zenodo.org/record/3384388*

### Build baseline anomaly detection model using reconstruction techniques
To reproduce the research use case, I built a baseline anomaly detection model using a 1-D convolutional neural network (CNN). The model employs a reconstruction technique. The reconstruction technique involves training a model using `normal data` and then using it to reconstruct new data points. The degree of difference between `the new data points` (reconstructed data points) and the `original data` (normal data) can be used to detect anomalies. Below is examples of signal spectrums under `normal` conditions.

{% include aligner.html images="portfolio/Anomaly/S2.png"%}

For interested readers, you can also download the dataset using the links above and follow my tutorials below. 

#### Table of Contents:
- 1. DataLoader Construction: A PyTorch class designed to efficiently load and preprocess sound-anomaly pairs.
- 2. Model Training: Optimization of the model using reconstruction loss, with the Adam optimizer ensuring efficient convergence.
- 3. Model Building: A structured 1D convolutional neural network (CNN) composed of stacked convolution layers to capture sound data patterns.
- 4. Model Evaluation: Performance assessment using the ROC score to evaluate the model's anomaly detection capabilities.


To manage the MIMII Dataset, I developed a custom PyTorch dataset class that loads (sound, anomaly) pairs.
- Sound Folder: This folder contains all the WAV audio files, organized for training, validation, and testing.
- label.csv: A key file that includes three columns: `file_id` (The unique identifier corresponding to each audio file in the sound folder),
`abnormal` (A binary flag indicating whether a sound is abnormal (1) or normal (0), `split` (Specifies the dataset split—whether the observation belongs to the training, validation, or testing set)

```python
class AUDIO(Dataset):
    def __init__(self,
                 sound_folder,
                 label_file,
                 split = "train", # train | valid | test
                 normalize = True,
                ):
        
        _label_df = pd.read_csv(label_file)
        self.label_df = _label_df[_label_df["split"] == split]
        self.sound_folder = sound_folder
        self.normalize = normalize
        
    def __len__(self):        
        return len(self.label_df)
        
    def __getitem__(self, index):
                
        _, abnormal, file_id = self.label_df.iloc[index].to_list()        
        
        # load sound
        signal, sr = torchaudio.load(f"{self.sound_folder}/{file_id}.wav")
        
        if self.normalize :
            signal = (signal-signal.min())/(signal.max()-signal.min())

        return torch.tensor(signal), torch.tensor(abnormal), sr, file_id
```

Next, to validate the dataset class, I loaded 5 normal sound signals from the validation set. You can hit the play button to hear and compare the sound of both normal and abnormal signals. 

```python
def visualize(dataset, 
             n_sample = 5):
    
    indexes = np.random.choice(range(len(dataset)), size = n_sample, replace = False)
    
    for index in indexes:
        
        signal, abnormal, sr, file_id = dataset[index]
        print(f"file_id:  {file_id}")
        print(f"abnormal: {abnormal}")
        
        ipd.display(ipd.Audio(np.array(signal[0]), rate = sr)) 


_dataset = AUDIO("sounds/sounds",
                 "label.csv",
                 split = "valid",
                 normalize = False)

visualize(_dataset)
```

Now, we construct 1D convolutional neural netwrok (CNN) to build a reconstruction model. One caveat in constructing the CNN architecture is that the input-size and output-size must match, and to this end, a decoder block should be added that maps the encoded signal back to its original size.

```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1, dilation=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),            
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1, dilation=4),
            nn.BatchNorm1d(64),            
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1, dilation=8),
            nn.BatchNorm1d(128),                        
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1, dilation=16),
            nn.BatchNorm1d(256),            
            nn.ReLU()
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(128),            
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=1, padding=1, dilation=8),
            nn.BatchNorm1d(64),            
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=4),
            nn.BatchNorm1d(32),            
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=1, padding=1, dilation=2),
            nn.BatchNorm1d(16),            
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        
    def forward(self, x):
        # Encode
        z = self.encoder(x)
        
        # Decode
        y = self.decoder(z)
        
        y = nn.functional.upsample(y, size=x.shape[-1])

        return y
```

```python
model = CNN().cuda()
```

For loss function, I used mean squared error (MSE) loss to measure the similarity between the original and reconstructed signals. The Adam optimizer was chosen for its efficiency and faster convergence compared to standard gradient descent methods.

```python
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
```

The model was trained for 10 epochs. Extending the training period mostly likely provide even better results. During each epoch, the mean squared error (MSE) score was calculated on the training set to monitor reconstruction performance. Additionally, the AUC score was evaluated on the validation set to track the model’s anomaly detection capabilities. The batch size was adjusted to prevent Out-Of-Memory (OOM) errors.

```python
def train(trainloader, validloader, model,
          n_epoch = 5):
    
    for epoch in range(n_epoch):
        print("")
        model.train()
        train_loss = train_epoch(trainloader, model)        
        print(f"Epoch {epoch+1}/{n_epoch}, Train MSE: {train_loss}")
        
        with torch.no_grad():    
            valid_dice = evaluate_epoch(validloader, model)     
            print(f"Epoch {epoch+1}/{n_epoch}, Valid AUC : {valid_dice}")
        
    return model

def train_epoch(trainloader, model):
        
    losses = []
    for (inputs, *_) in trainloader:
        # forward pass  
        inputs = inputs.cuda()        
        outputs = model(inputs) 
        
        # calculate loss
        loss = loss_fn(outputs, inputs)

        # backward pass and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    
    return np.mean(losses)

def evaluate_epoch(validloader, model):
    
    scores = []
    labels = []
    for (inputs, targets, *_)  in validloader:
        
        inputs = inputs.cuda()
        
        outputs = model(inputs)
                        
        scores.append(outputs.detach().cpu().numpy().sum())
        labels.append(targets.detach().cpu().numpy().squeeze())
        
    auc = roc_auc_score(np.array(labels),
                        np.array(scores))
    
    return auc
```

```python
sound_folder  = "sounds/sounds"
label_file = "label.csv"

train_dataset = AUDIO(sound_folder,
                      label_file,
                      split = "train")

valid_dataset = AUDIO(sound_folder,
                      label_file,
                      split = "valid")

train_loader = DataLoader(train_dataset,
                          batch_size  = 32,
                          num_workers = 8,
                          shuffle     = True, 
                          pin_memory  = True)

valid_loader = DataLoader(valid_dataset,
                          batch_size  = 1,
                          num_workers = 1,
                          shuffle     = False,
                          pin_memory  = False)

model = train(train_loader, valid_loader, model,
              n_epoch = 10)
```

Now we can get the prediction score on the test set
```python
def predict(model, loader):
    
    test_results = []
    for (inputs, *_, file_id) in loader:
        
        inputs = inputs.cuda()
        
        outputs = model(inputs)
                        
        score = outputs.detach().cpu().numpy().sum()
        
        test_results.append({"file_id" : file_id.item(),
                             "score"  : score})
        
    return test_results
```

```python
sound_folder  = "sounds/sounds"
label_file = "label.csv"

test_dataset = AUDIO(sound_folder,
                     label_file,
                     split = "test")

test_loader = DataLoader(test_dataset,
                         batch_size  = 1,
                         num_workers = 1,
                         shuffle     = False, 
                         pin_memory  = False)
    
test_results = predict(model, test_loader)

df_final = pd.DataFrame.from_dict(test_results)

df_final.to_csv("df_final.csv", index = False)
```


Here's the end-to-end Python scripts.
```python
import random, os, re, sys

import numpy as np
import pandas as pd

from toolz import *
from toolz.curried import *

from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score
    
import matplotlib.pyplot as plt
import IPython.display as ipd
import librosa.display

import warnings
warnings.filterwarnings("ignore")

torch.multiprocessing.set_sharing_strategy('file_system')

class AUDIO(Dataset):
    def __init__(self,
                 sound_folder,
                 label_file,
                 split = "train", # train | valid | test
                 normalize = True,
                ):
        
        _label_df = pd.read_csv(label_file)
        self.label_df = _label_df[_label_df["split"] == split]
        self.sound_folder = sound_folder
        self.normalize = normalize
        
    def __len__(self):        
        return len(self.label_df)
        
    def __getitem__(self, index):
                
        _, abnormal, file_id = self.label_df.iloc[index].to_list()        
        
        # load sound
        signal, sr = torchaudio.load(f"{self.sound_folder}/{file_id}.wav")
        
        if self.normalize :
            signal = (signal-signal.min())/(signal.max()-signal.min())

        return torch.tensor(signal), torch.tensor(abnormal), sr, file_id

def visualize(dataset, 
             n_sample = 5):
    
    indexes = np.random.choice(range(len(dataset)), size = n_sample, replace = False)
    
    for index in indexes:
        
        signal, abnormal, sr, file_id = dataset[index]
        print(f"file_id:  {file_id}")
        print(f"abnormal: {abnormal}")
        
        ipd.display(ipd.Audio(np.array(signal[0]), rate = sr)) 


_dataset = AUDIO("sounds/sounds",
                 "label.csv",
                 split = "valid",
                 normalize = False)

visualize(_dataset)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1, dilation=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),            
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1, dilation=4),
            nn.BatchNorm1d(64),            
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1, dilation=8),
            nn.BatchNorm1d(128),                        
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1, dilation=16),
            nn.BatchNorm1d(256),            
            nn.ReLU()
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(128),            
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=1, padding=1, dilation=8),
            nn.BatchNorm1d(64),            
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=4),
            nn.BatchNorm1d(32),            
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=1, padding=1, dilation=2),
            nn.BatchNorm1d(16),            
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        
    def forward(self, x):
        # Encode
        z = self.encoder(x)
        
        # Decode
        y = self.decoder(z)
        
        y = nn.functional.upsample(y, size=x.shape[-1])

        return y

model = CNN().cuda()

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def train(trainloader, validloader, model,
          n_epoch = 5):
    
    for epoch in range(n_epoch):
        print("")
        model.train()
        train_loss = train_epoch(trainloader, model)        
        print(f"Epoch {epoch+1}/{n_epoch}, Train MSE: {train_loss}")
        
        with torch.no_grad():    
            valid_dice = evaluate_epoch(validloader, model)     
            print(f"Epoch {epoch+1}/{n_epoch}, Valid AUC : {valid_dice}")
        
    return model

def train_epoch(trainloader, model):
        
    losses = []
    for (inputs, *_) in trainloader:
        # forward pass  
        inputs = inputs.cuda()        
        outputs = model(inputs) 
        
        # calculate loss
        loss = loss_fn(outputs, inputs)

        # backward pass and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    
    return np.mean(losses)

def evaluate_epoch(validloader, model):
    
    scores = []
    labels = []
    for (inputs, targets, *_)  in validloader:
        
        inputs = inputs.cuda()
        
        outputs = model(inputs)
                        
        scores.append(outputs.detach().cpu().numpy().sum())
        labels.append(targets.detach().cpu().numpy().squeeze())
        
    auc = roc_auc_score(np.array(labels),
                        np.array(scores))
    
    return auc

sound_folder  = "sounds/sounds"
label_file = "label.csv"

train_dataset = AUDIO(sound_folder,
                      label_file,
                      split = "train")

valid_dataset = AUDIO(sound_folder,
                      label_file,
                      split = "valid")

train_loader = DataLoader(train_dataset,
                          batch_size  = 32,
                          num_workers = 8,
                          shuffle     = True, 
                          pin_memory  = True)

valid_loader = DataLoader(valid_dataset,
                          batch_size  = 1,
                          num_workers = 1,
                          shuffle     = False,
                          pin_memory  = False)

model = train(train_loader, valid_loader, model,
              n_epoch = 10)

def predict(model, loader):
    
    test_results = []
    for (inputs, *_, file_id) in loader:
        
        inputs = inputs.cuda()
        
        outputs = model(inputs)
                        
        score = outputs.detach().cpu().numpy().sum()
        
        test_results.append({"file_id" : file_id.item(),
                             "score"  : score})
        
    return test_results

sound_folder  = "sounds/sounds"
label_file = "label.csv"

test_dataset = AUDIO(sound_folder,
                     label_file,
                     split = "test")

test_loader = DataLoader(test_dataset,
                         batch_size  = 1,
                         num_workers = 1,
                         shuffle     = False, 
                         pin_memory  = False)
    
test_results = predict(model, test_loader)

df_final = pd.DataFrame.from_dict(test_results)

df_final.to_csv("df_final.csv", index = False)
```
from imutils import paths
from torch.utils.data import Dataset
from PIL import Image
from random import randrange
import wordCounter as wc
import pandas as pd
import numpy as np
import random

class FlickrTrainDataset(Dataset):
    def __init__(self,path_images,path_captions,transform=None,type=None):
        
        self.paths = list(paths.list_images(path_images))
        self.paths.sort()
        self.captions = pd.read_csv(path_captions,sep='|')[' comment']
        
        ## Split the dataset in train and validation
        ## Sample without replacement the 99% of the total images to take train images
        indexes_of_train = random.sample(range(len(self.paths)),int(0.99*len(self.paths)))
        
        if type=='TRAIN':
            self.paths = [self.paths[i] for i in indexes_of_train]
            self.captions = [self.captions[i] for i in indexes_of_train]
        else:
            self.paths = [self.paths[i] for i in self.paths if i not in indexes_of_train]
            self.captions = [self.captions[i] for i in self.captions if i not in indexes_of_train]
            return
        
        self.word_to_idx = wc.wordToIdx(path_captions,length=24)
        self.transform = transform


    def __len__(self):
        return len(paths)

    def getVocabSize(self):
        return self.word_to_idx.vocabSize()

    def getMaxCaptionsLength(self):
        return self.word_to_idx.maxCaptionLength()

    def __getitem__(self, item):
        x = self.paths[item]
        x = Image.open(x)
        if self.transform is not None:
            x = self.transform(x)

        caps = self.captions[5*item:5*(item+1)]
        y,z = self.word_to_idx.captionsToTensors(caps)

        # Pick one of the captions for 1-1 image-caption training
        r=randrange(5)

        y=np.array(y[r])
        z=np.array(z[r])

        return x,y,z







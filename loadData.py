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
            self.paths = self.paths[indexes_of_train]
            self.captions = self.captions[indexes_of_train]
        else:
            
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

        r=randrange(5)

        y=np.array(y[r])
        z=np.array(z[r])

        return x,y,z


def trainer(model,optimizer,loader_train,epochs=1):
    print_every=3
    for e in range(epochs):
        for t ,(x,y) in enumerate(loader_train):
            #model.train()  # put model to training mode
            #x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            #y = y.to(device=device, dtype=torch.long)

            #scores = model(x)
            #loss = F.cross_entropy(scores, y)


            #optimizer.zero_grad()

            # loss.backward() computes dloss/dx for every parameter x which has requires_grad=True.
            #loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            #optimizer.step()

            if t % print_every == 0:
                print('Iteration %d, loss =',t," ",x)




from imutils import paths
from torch.utils.data import Dataset
from PIL import Image
import wordCounter as wc
import pandas as pd
import numpy as np

class FlickrTrainDataset(Dataset):
    def __init__(self,path_images,path_captions,transform=None):
        self.paths = list(paths.list_images(path_images))
        self.paths.sort()
        self.word_to_idx = wc.wordToIdx(path_captions,length=24)
        self.captions = pd.read_csv(path_captions,sep='|')[' comment']
        self.transform = transform

        self.paths = self.paths[:10]
        self.captions = self.captions[:50]
    def __len__(self):
        return 10

    def __getitem__(self, item):
        x = self.paths[item]
        x = Image.open(x)
        if self.transform is not None:
            x = self.transform(x)

        caps = self.captions[5*item:5*(item+1)]
        y,z = self.word_to_idx.captionsToTensors(caps)

        y=np.array(y)
        z=np.array(z)

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




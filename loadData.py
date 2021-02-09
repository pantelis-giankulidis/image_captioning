from imutils import paths
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms

class FlickrTrainDataset(Dataset):
    def __init__(self,path,transform=None):
        self.paths = list(paths.list_images('../flickr/flickr30k_images/flickr30k_images'))
        self.paths.sort()
        self.captions = pd.read_csv('../flickr/flickr30k_images/results.csv',sep='|')
        self.transform = transform

    def __len__(self):
        return 50

    def __getitem__(self, item):
        x = self.paths[item]
        x = Image.open(x)
        if self.transform is not None:
            x = self.transform(x)
        y = self.captions[5*item:5*(item+1)]
        return x,y


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


trans = transforms.Compose([transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train = FlickrTrainDataset('hi',trans)
loader_train = DataLoader(train,32,sampler=sampler.SubsetRandomSampler(range(10000)))
trainer(0,0,loader_train)


from imutils import paths
from torch.utils.data import Dataset
from PIL import Image
import wordCounter as wc

class FlickrTrainDataset(Dataset):
    def __init__(self,path_images,path_captions,transform=None):
        self.paths = list(paths.list_images(path_images))
        self.paths.sort()
        self.word_to_idx,self.cap_lengths = wc.wordToIdx(path_captions)

        self.transform = transform

    def __len__(self):
        return 50

    def __getitem__(self, item):
        x = self.paths[item]
        x = Image.open(x)
        if self.transform is not None:
            x = self.transform(x)
        caps = self.captions[5*item:5*(item+1)]
        y=list()
        i=0
        for sentence in caps:
            if(isinstance(sentence,float)):
                continue
            y.append(list())
            for word in sentence:
                y[i].append(self.word_to_idx[word.lower()])
            i=i+1

        return x,y,self.cap_lengths[item]


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




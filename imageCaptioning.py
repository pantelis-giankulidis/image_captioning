from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torch
import torchvision.transforms as transforms
import train
import model
import loadData
import pickle

device = torch.device("cpu")

# Pretrained model in Imagenet has mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]
# For models trained with other datasets,we should use different
# modification

trans = transforms.Compose([transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
     transforms.Resize((128,128))])


trainDataset = loadData.FlickrTrainDataset('../flickr/flickr30k_images/flickr30k_images','../flickr/flickr30k_images/results.csv',trans)
voc_size = trainDataset.getVocabSize()
max_capt = trainDataset.getMaxCaptionsLength()

<<<<<<< HEAD

=======
Encoder = model.Encoder()
Decoder = model.Decoder(encoder_dim=2048,decoder_dim=512,attention_dim=256,vocab_size-voc_size)
Embedding = model.Embedding(vocab_size=voc_size,embedding_dim=128)

loader_train = DataLoader(trainDataset,32,sampler=sampler.SubsetRandomSampler(range(32)))
train.train(data_loader=loader_train,encoder=Encoder,decoder=Decoder,embedding=Embedding,max_caption_length=max_capt)


with open('encoder-5',"wb") as f:
     pickle.dump(Encoder,f)
     
with open('decoder-5',"wb") as f:
     pickle.dump(Decoder,f)
     
with open('vocab',"wb") as f:
     pickle.dump(Embedding,f)
>>>>>>> 01f2f3fdcbe62dc48696f1b6d08e5c4d94dfcc54

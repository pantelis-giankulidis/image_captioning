from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torch
import torchvision.transforms as transforms
import train
import model
import loadData
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pretrained model in Imagenet has mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]
# For models trained with other datasets,we should use different
# modification

### Epochs of training
epochs = 5

### The destination where the raw images from flickr30k dataset are stored
images_folder = '../flickr/flickr30k_images/flickr30k_images'

### The csv conatining the corresponding captions
captions_folder = '../flickr/flickr30k_images/results.csv'

trans = transforms.Compose([transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
     transforms.Resize((128,128))])

def main():
     ### Create the torch datasets and get the size of the 'on-the-fly' created vocabulary and the length of the longest caption
     trainDataset = loadData.FlickrTrainDataset(images_folder,captions_folder,trans,'TRAIN')
     valDataset = loadData.FlickrValDataset(images_folder,captions_folder,trans,'VAL')
     voc_size = trainDataset.getVocabSize()
     max_capt = trainDataset.getMaxCaptionsLength()

     ### Create the models
     Encoder = model.Encoder()
     Decoder = model.Decoder(encoder_dim=2048,decoder_dim=512,attention_dim=256,vocab_size=voc_size)
     Embedding = model.Embedding(vocab_size=voc_size,embedding_dim=128)

     ### Set the optimizer for the decoder(the only component that is actually trained) and the device for the model tensors
     decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, Decoder.parameters()),lr=e-3)
     Encoder.to(device)
     Decoder.to(device)
     Embedding.to(device)
     
     ### Create the data loaders for training and evaluation
     loader_train = DataLoader(trainDataset,32,sampler=sampler.SubsetRandomSampler(range(30000)))
     val_loader= DataLoader(valDataset,32,sampler=sampler.SubsetRandomSampler(range(30000)))
     
     best_bleu = 0 #The best blue score by now
     for i in range(epochs):
          ## One epoch's training
          train.train(data_loader=loader_train,encoder=Encoder,decoder=Decoder,embedding=Embedding,max_caption_length=max_capt,optim=decoder_optimizer)
          ## One epoch's validation
          new_bleu = train.validate(data_loader=val_loader,encoder=Encoder,decoder=Decoder,embedding=Embedding,max_capt)
          
          if new_bleu > best_bleu:
               best_bleu=new_bleu
          else:
               ## We had no improvement since last time,so se don't train more
               break
      
     ## Save the model for deploying
     torch.save(Encoder,'Encoder')
     torch.save(Decoder,'Decoder')
     torch.save(Embedding,'Embedding')
     
if __name__ == '__main__':
     main()

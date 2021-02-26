from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torch
import torchvision.transforms as transforms
import train
import model
import loadData

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

#Define the models
encoder = model.Encoder()
decoder = model.Decoder(encoder_dim=2048,decoder_dim=512,attention_dim=256,vocab_size=voc_size)
embedding=model.Embedding(vocab_size=voc_size,embedding_dim=128)

loader_train = DataLoader(trainDataset,32,sampler=sampler.SubsetRandomSampler(range(32)))
train.train(data_loader=loader_train,encoder=encoder,decoder=decoder,embedding=embedding,max_caption_length=max_capt)


modelCnn_json = encoder.to_json()
modelRnn_json = decoder.to_json()
modelEmb = embedding.to_json()

with open("model.json", "w") as json_file:
    modelCnn_json.write(modelCnn_json)


# serialize weights to HDF5
encoder.save_weights("model.h5")

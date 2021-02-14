from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.transforms as transforms
import train
import model
import loadData as ld



# Pretrained model in Imagenet has mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]
# For models trained with other datasets,we should use different
# modification

trans = transforms.Compose([transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
     transforms.Resize((128,128))])


trainDataset = ld.FlickrTrainDataset('../flickr/flickr30k_images/flickr30k_images','../flickr/flickr30k_images/results.csv',trans)
loader_train = DataLoader(trainDataset,8,sampler=sampler.SubsetRandomSampler(range(8)))
train.train(data_loader=loader_train,encoder=model.Encoder())
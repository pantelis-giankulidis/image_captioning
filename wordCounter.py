import pandas as pd
import numpy as np

greater = lambda a,b: a if a>b else b

class wordToIdx():

    def __init__(self,path,length):
        captions = list(pd.read_csv(path, sep='|')[' comment'])
        self.word_to_idx=dict()
        id=4

        self.word_to_idx['<<START>>']=0
        self.word_to_idx['<<END>>']=1
        self.word_to_idx['<<SPACE>>']=2
        self.word_to_idx['<<UNKNOWN>>']=3

        maxCaptionLength=0

        for sentence in captions:
            if(isinstance(sentence,float)):
                continue
            length=0
            for word in sentence.split(" "):
                length=length+1
                if(word.lower() not in self.word_to_idx.keys()):
                    self.word_to_idx[word.lower()]=id
                    id=id+1
            maxCaptionLength = greater(maxCaptionLength,length)

        self.captionsLength=maxCaptionLength
        self.vocab_size = id

    def vocabSize(self):
        return self.vocab_size

    def maxCaptionLength(self):
        return self.captionsLength

    def captionsToTensors(self,captions):

        x=list()
        y=list()
        i=0
        for sentence in captions:
            word_counter=0
            if (isinstance(sentence, float)):
                continue
            x.append(list()) # List to store the new caption indexes
            x[i].append(0) # Store the <<START>> character
            for word in sentence.split(" "):
                word_counter=word_counter+1
                if(word.lower() not in self.word_to_idx.keys()):
                    x[i].append(3) # Store <<UNKNOWN>> character
                else:
                    x[i].append(self.word_to_idx[word.lower()])

            x[i].append(1) # Store the <<END>> character
            y.append(word_counter) # Store the quantity of words in the caption

            while(len(x[i])<self.captionsLength):
                x[i].append(2) # Fill the empty spaces

            i =i+1
        return x,y

'''caps = pd.read_csv('../flickr/flickr30k_images/results.csv',sep='|')[" comment"]

wi = wordToIdx('../flickr/flickr30k_images/results.csv',24)

indexes,lengths = wi.captionsToTensors(caps)'''



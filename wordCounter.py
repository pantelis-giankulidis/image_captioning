import pandas as pd

class wordToIdx():

    def __init__(self,path,length):
        captions = list(pd.read_csv(path, sep='|')[' comment'])
        i=0
        self.word_to_idx=dict()
        self.sentence_length=list()
        id=4

        self.word_to_idx['<<START>>']=0
        self.word_to_idx['<<END>>']=1
        self.word_to_idx['<<SPACE>>']=2
        self.word_to_idx['<<UNKNOWN>>']=3

        for sentence in captions:
            word_counter=0
            if(isinstance(sentence,float)):
                continue

            for word in sentence.split(" "):
                if(word.lower() not in self.word_to_idx.keys()):
                    self.word_to_idx[word.lower()]=id
                    id=id+1

                word_counter=word_counter+1
            i=i+1
            self.sentence_length.append(word_counter)

        self.captionsLength=length

    def captionsToTensors(self,captions):

        x=list()
        i=0
        for sentence in captions:
            if (isinstance(sentence, float)):
                continue
            x.append(list()) # List to store the new caption indexes
            x[i].append(0) # Store the <<START>> character
            for word in sentence.split(" "):
                if(word.lower() not in self.word_to_idx.keys()):
                    x[i].append(3)
                else:
                    x[i].append(self.word_to_idx[word.lower()])
            x[i].append(1) # Store the <<END>> character

            while(len(x[i])<self.captionsLength):
                x[i].append(2) # Fill the empty spaces

            i =i+1
        return x


c = wordToIdx('../flickr/flickr30k_images/results.csv',20)
idxs = c.captionsToTensors(['a man sits in the beach.','a woman and a man sit on the beach'])
print(idxs)

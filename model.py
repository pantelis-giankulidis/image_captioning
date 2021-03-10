from torch import nn
import torchvision.models as cnn
import torch

class Encoder(nn.Module):
    def __init__(self,encoded_image_size=16):
        super(Encoder, self).__init__()

        resnet = cnn.resnet152(pretrained=True)

        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune(False)

    def forward(self, images):
        out = self.resnet(images)
        out = self.adaptive_pool(out)
        
        return out #(batch_size,H,W,C)

   


class Attention(nn.Module):
    def __init__(self, encoder_dim,decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights#Soft or Hard?

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, of dimension (N x encoder_dim x (HxW))
        :param decoder_hidden: previous decoder output, a of dimension (N x decoder_dim)
        :return: attention weighted encoding, weights
        """
        encoder_out = encoder_out.permute(0,-1,1)
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)

        #TODO: Look for other version of attention
        attention_weights = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weights,alpha

class Embedding(nn.Module):
    def __init__(self,vocab_size,embedding_dim):
        super(Embedding, self).__init__()
        self.embeding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self,captions):
        return self.embeding(captions)

class Decoder(nn.Module):


    def __init__(self, encoder_dim, decoder_dim, attention_dim,vocab_size, alpha_c=1.01,dropout=0.2):
        super(Decoder,self).__init__()
        self.initial_c = nn.Linear(encoder_dim,decoder_dim)
        self.initial_h = nn.Linear(encoder_dim,decoder_dim)

        self.dropout = nn.Dropout(dropout)
        self.step = nn.LSTMCell(encoder_dim+128,decoder_dim,bias=True)

        self.sigmoid = nn.Linear(decoder_dim,encoder_dim) # recursive h

        self.final_scores = nn.Linear(decoder_dim,vocab_size) # scores to vocabulary words
        self.Attention = Attention(encoder_dim,decoder_dim,attention_dim)
        self.Embedding = Embedding(vocab_size,256)
        self.vocab_size = vocab_size
        self.fc = nn.Linear(decoder_dim,vocab_size)

        self.alpha_c = alpha_c

    def forward(self,encoded_image,embeddings,caption_lengths,num_pixels,n=32):

        """

        :param encoded_image: The encoded image of size (encoder_dim x H x W)
        :param embeddings: The caption embeddings of size (max_caption_length x D)

        :return:
        """

        mean_encoder = encoded_image.mean(dim=2) #Mean value of pixels in the encoded image
        #TODO: Search for proper initialization of c,h
        h = self.initial_h(mean_encoder)
        c = self.initial_c(mean_encoder)

        decode_lens = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros([n, max(decode_lens), self.vocab_size])
        alphas = torch.zeros([n, max(decode_lens), num_pixels])
        #embeddings = embeddings.view(embeddings.size(0)*embeddings.size(1),embeddings.size(2),embeddings.size(3))

        for t in range(max(decode_lens)):
            #batch_size_t = sum([l > t for l in decode_lens])
            attention_weighted_encoding, alpha = self.Attention(encoded_image,h)

            gate = self.sigmoid(h)  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding


            h, c = self.step(torch.cat([embeddings[ :, t, :], attention_weighted_encoding], dim=1),(h, c))  # (batch_size_t, decoder_dim)

            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:, t, :] = preds
            alphas[:, t, :] = alpha

        return preds,self.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

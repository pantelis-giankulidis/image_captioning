from torch import nn
import torch

decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),lr=0.01)

print_freq = 2

def train(data_loader,encoder,decoder,embedding,max_caption_length,epochs=1):

    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()), lr=0.01)
    encoder.train() #Train mode for CNN
    decoder.train() #Train mode for LSTM
    embedding.train()

    for e in range(epochs):
        for i, (image,captions,caption_lengths) in enumerate(data_loader):

            images = encoder(image)
            emb= embedding(captions) #(N,max_caption_length,embedding_dim)

            images = images.view(images.size(0),images.size(1),images.size(2)*images.size(3))

            scores,stochastic_loss =decoder(images,emb,caption_lengths,num_pixels=256)

            targets = captions[:, 1:]#targets are from the first prediction

            loss = nn.CrossEntropyLoss(scores, targets)
            loss+=stochastic_loss

        	# Add doubly stochastic attention regularization
            #loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Back prop.
            decoder_optimizer.zero_grad()
            
            loss.backward()

        	# Update weights
            decoder_optimizer.step()
            

            # Print status
            if i % print_freq == 0:
                print('Epoch ',i,' loss: %.2f'.format(loss))
            if i==2:
                break

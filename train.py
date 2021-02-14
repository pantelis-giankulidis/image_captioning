def train(data_loader,encoder,decoder=None,epochs=1):
    for e in range(epochs):
        for t, (image,captions,caption_lengths) in enumerate(data_loader):
            #images = encoder(image)
            print("Iteration ",t," , image ",image,"   capt ",captions,"  length  ",caption_lengths)




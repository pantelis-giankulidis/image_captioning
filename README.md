# image_captioning
## Introduction
This is an image captioning model using Attention mechanism.The idea comes from the classic paper <a href="https://arxiv.org/abs/1502.03044">[1]</a> although many alternations and different hyperparameters are used.For a very nice introduction of the theory of this image captioning you can see <a href="https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning">here</a>.Hereby,the focus will be on the improvements that were proposed on the aforementioned work,as well as,their justification. 

## Architecture
The project consists of three main components,an encoder,that takes the image as input and generates a Tensor,a decoder that takes as input the Tensor and an embedding that converts the numerical presentation of the captions into lower dimenions.
#### Encoder

The encoder takes an RGB image as input and generates a new **HxWxC** tensor,where the C channels contain each a different summary representation of the initial image.To do this "compression" and "filtering" of the information of the image,we use a pretrained convolutional neural network.More specific,the torchvision Residual network 152 is used,one of the best models for captioning high dimension relations in an image. 


![data-original](https://user-images.githubusercontent.com/67536962/109940644-faf4e400-7cda-11eb-84ad-7bee14aa83f7.png)

Because the model isn't used for image classification,the last fully connected layer is replaced by a new convolution layer of C windows , in order for the encoder to output the desired tensor.

#### Decoder
The decoder is an Long short term memory (LSTM) recurrent neural network (RNN) that generates a caption given the representation of the image created by the Encoder and the caption from the previous LSTM cell,after it gets through the attention mechanism and the embedding layer.

![Unt](https://user-images.githubusercontent.com/67536962/110251298-8a72ef00-7f88-11eb-8308-648e798d77b1.jpg)




One LSTM cell,tries to generate the next word by taking into consideration three aspects;the parameters of the "previous" cell(in fact we have only multiple instances of one cell),the area in the image where it must focus(which is the output of the attention mechanism) and the embedding of the word that has been generated last(the output of the embedding)

#### Attention 
The architecture here is novel.It follows some basic concepts from previous implementations(see the links below),but is a brand new.It has three linear layers for more independent parameter learning,instead of the traditional two.This way,both the image region and the embedding are attentioned before used as input to the decoder.In both cases,soft attention is used.In this type of attention,the weights of the weights sum to 1.In the image,the weights represent the propability
that the pixel is the place to look when generating the next word.In the embedding,the weights represent the importance of the previous word to the generation of the next word.This makes the captions more syntactically correct.

#### Embedding
We use a pretrained torch embedding of 256 size.
`nn.Embedding(n,256)`
It may seem small,but it achieves adequate "compression" of the words.There is also the option to fine tune the embedding but this opstion wasn't used in the experiments. 

## Training
For training, the popular flickr30k dataset available in <a href="https://www.kaggle.com/hsankesara/flickr-image-dataset">kaggle</a> was used.A torch `Dataset`
was created and a torch `Dataloader` of batch size 32.The loss function that was used is the **crossEntropyLoss**.The raw scores from the final layer of the decoder are submmited and compared,using the aforementioned loss function with the actual captions.The feedback is than propagated to the LSTM to be trained.
As it is already mentioned,the encoder is already trained,as well as the embedding.
It is worth noticing,that the writers of the original paper,suggest the use of the **double stochastic regularization** loss.In the provided git there is a simple explanation about that.

## Performance
The model was trained with the flickr30k dataset.To assess its performance,we use the BLUE-1(bilingual evaluation understudy),a popular choice for NLP model evaluation.I didn't run many experiments, but the results are presented in the array below. Only the decoder was trained in all the experiments,with the same hyperparameters(see source code) in each case.
| Training epochs | Validation BLEU-1 score | Test BLEU-1 score |
| ----------------|-------------------------|-------------------|
| 1               |  18.85                  | 16.91             |
| 3               |  18.44                  | 18.16             |
| 4               |  18.91                  | 18.49             |
| 5               |  20.78                  | 18.90             |

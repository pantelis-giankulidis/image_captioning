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
![Diagram1](https://user-images.githubusercontent.com/67536962/110094371-6a82d600-7da4-11eb-8403-65461dbd4a20.png)


One LSTM cell,tries to generate the next word by taking into consideration three aspects;the parameters of the "previous" cell(in fact we have only multiple instances of one cell),the area in the image where it must focus(which is the output of the attention mechanism) and the embedding of the word that has been generated last(the output of the embedding)

#### Attention 
The architecture here is novel.It follows some basic concepts from previous implementations(see the links below),but is a brand new.

#### Embedding
We use a pretrained embedding of 256 size.It may seem small,but it achieves adequate "compression" of the words. 

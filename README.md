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

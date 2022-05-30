# Convolutional Neural Network
Different types of convolutional neural networks have been implemented using Keras API of Tensorflow in colab notebook. Also, the basic convolution and maxpooling operation on an input matrix using a kernel matrix has been demonstrated.

# Regular CNN:
A regular CNN is a CNN where the number of filters in each layer increases as the depth of the network grows i.e., the Lth layer will have more filters than the (L-1)th layer.A  regular CNN has been created using 10 convolutional layers. The output size was kept same as input of each layer by using padding = ‘same’. A 2 by 2 maxpooling with stride 2 has been used after each five convolutional layers. The code is given in the colab notebook file (**regular_CNN.ipynb**). Initially, we used ‘adam’ optimizer, batch size 64, number of epochs 3.

The training and test loss and accuracy are reported as below. Also, in colab notebook, tensorboard graphs are presented.

Epoch 1/3

938/938 [==============================] - 111s 117ms/step - loss: 0.6786 - accuracy: 0.7556

Epoch 2/3

938/938 [==============================] - 109s 116ms/step - loss: 0.1157 - accuracy: 0.9641

Epoch 3/3

938/938 [==============================] - 110s 117ms/step - loss: 0.0804 - accuracy: 0.9752

313/313 [==============================] - 8s 27ms/step - loss: 79.4283 - accuracy: 0.4665

Test loss: 79.42828369140625

Test accuracy: 0.46650001406669617

It is noticed that the accuracy increases with increasing number of epochs, but it takes longer time to train the model too. To see the effect of batch size on the performance of the model, we chose epochs = 1 and compared the achieved training and testing accuracies in below table:

![alt text](https://github.com/Zobaer/ConvolutionalNeuralNet/blob/main/figs/Table1.png)

Later, we tried SGD and RMSprop optimizers with epochs = 3 and batch size = 64. The final training and test loss and accuracies are presented below:


SGD:
Epoch 1/3

938/938 [==============================] - 110s 116ms/step - loss: 2.2932 - accuracy: 0.1208

Epoch 2/3

938/938 [==============================] - 109s 116ms/step - loss: 0.7858 - accuracy: 0.7438

Epoch 3/3

938/938 [==============================] - 109s 116ms/step - loss: 0.3139 - accuracy: 0.9010

313/313 [==============================] - 9s 27ms/step - loss: 193.1294 - accuracy: 0.6914

Test loss: 193.12940979003906

Test accuracy: 0.6913999915122986


RMSPROP:

Epoch 1/3

938/938 [==============================] - 112s 118ms/step - loss: 0.3738 - accuracy: 0.8764

Epoch 2/3

938/938 [==============================] - 109s 116ms/step - loss: 0.1064 - accuracy: 0.9661

Epoch 3/3

938/938 [==============================] - 109s 116ms/step - loss: 0.0722 - accuracy: 0.9771

313/313 [==============================] - 9s 27ms/step - loss: 7.4484 - accuracy: 0.9798

Test loss: 7.448370456695557

Test accuracy: 0.9797999858856201

![alt text](https://github.com/Zobaer/ConvolutionalNeuralNet/blob/main/figs/Table2.png)

So, RMSProp is giving us the lowest loss and accuracy in both training and test data, which can further be improved by increasing batch size.

Now, let’s investigate the effect of learning rate on the model performance. We will change learning rate for RMSProp optimizer since it was giving the best result. The default learning rate was 0.001 in previous tests. We varied learning rate keeping batch size 64 and epochs = 1 (for faster training) and tabularized the results in below table:

![alt text](https://github.com/Zobaer/ConvolutionalNeuralNet/blob/main/figs/Table3.png)

So, the accuracy is increasing with decreasing learning rate within learning rate .001 to 1.

# Inverted CNN

The code is available in inverted_CNN.ipynb notebook file. With Adam optimizer, epochs = 3, and batch size = 512, below loss and accuracy have been achieved:

Epoch 1/3

118/118 [==============================] - 130s 1s/step - loss: 0.4360 - accuracy: 0.8619

Epoch 2/3

118/118 [==============================] - 130s 1s/step - loss: 0.2844 - accuracy: 0.9108

Epoch 3/3

118/118 [==============================] - 130s 1s/step - loss: 0.2259 - accuracy: 0.9283

313/313 [==============================] - 9s 29ms/step - loss: 34.1011 - accuracy: 0.9152

Test loss: 34.10105895996094

Test accuracy: 0.9151999950408936

The performance of this model is good but worse than regular CNN, as seen from the accuracy and loss values.

# Hourglass CNN:

The code is available in hourglass_CNN.ipynb notebook file.

With Adam optimizer, epochs = 3, and batch size = 512, below loss and accuracy have been achieved:

Epoch 1/3

118/118 [==============================] - 86s 721ms/step - loss: 2.3020 - accuracy: 0.1115

Epoch 2/3

118/118 [==============================] - 85s 721ms/step - loss: 2.3014 - accuracy: 0.1124

Epoch 3/3

118/118 [==============================] - 85s 718ms/step - loss: 2.3012 - accuracy: 0.1124

313/313 [==============================] - 11s 33ms/step - loss: 73.6535 - accuracy: 0.1033

Test loss: 73.65352630615234

Test accuracy: 0.10329999774694443

This model performance is worst among the three CNN models. It’s accurate in only 10% cases for both training and test data.

As a summary, to have an optimized performance, we will go ahead with regular CNN with RMSProp optimizer, batch size = 64, epochs = 3 (more can be better but computational time will increase significantly with little amount of improvement in accuracy) and learning rate 0.001.





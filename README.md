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

![alt text](https://github.com/Zobaer/BreathingAnomalyDetection/blob/main/figs/System%20model.png)

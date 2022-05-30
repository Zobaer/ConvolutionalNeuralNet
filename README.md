# Convolutional Neural Network
Different types of convolutional neural networks have been implemented using Keras API of Tensorflow in colab notebook. Also, the basic convolution and maxpooling operation on an input matrix using a kernel matrix has been demonstrated.

# Regular CNN:
A regular CNN is a CNN where the number of filters in each layer increases as the depth of the network grows i.e., the Lth layer will have more filters than the (L-1)th layer.A  regular CNN has been created using 10 convolutional layers. The output size was kept same as input of each layer by using padding = ‘same’. A 2 by 2 maxpooling with stride 2 has been used after each five convolutional layers. The code is given in the colab notebook file (**regular_CNN.ipynb**). Initially, we used ‘adam’ optimizer, batch size 64, number of epochs 3.

The training and test loss and accuracy are reported as below. Also, in colab notebook, tensorboard graphs are presented.

Epoch 1/3: 938/938 [==============================] - 111s 117ms/step - loss: 0.6786 - accuracy: 0.7556

Epoch 2/3: 938/938 [==============================] - 109s 116ms/step - loss: 0.1157 - accuracy: 0.9641

Epoch 3/3: 938/938 [==============================] - 110s 117ms/step - loss: 0.0804 - accuracy: 0.9752

313/313 [==============================] - 8s 27ms/step - loss: 79.4283 - accuracy: 0.4665

Test loss: 79.42828369140625

Test accuracy: 0.46650001406669617

It is noticed that the accuracy increases with increasing number of epochs, but it takes longer time to train the model too. To see the effect of batch size on the performance of the model, we chose epochs = 1 and compared the achieved training and testing accuracies in below table:

![alt text](https://github.com/Zobaer/ConvolutionalNeuralNet/blob/main/figs/Table1.png)

Later, we tried SGD and RMSprop optimizers with epochs = 3 and batch size = 64. The final training and test loss and accuracies are presented below:


SGD:
Epoch 1/3: 938/938 [==============================] - 110s 116ms/step - loss: 2.2932 - accuracy: 0.1208

Epoch 2/3: 938/938 [==============================] - 109s 116ms/step - loss: 0.7858 - accuracy: 0.7438

Epoch 3/3: 938/938 [==============================] - 109s 116ms/step - loss: 0.3139 - accuracy: 0.9010

313/313 [==============================] - 9s 27ms/step - loss: 193.1294 - accuracy: 0.6914

Test loss: 193.12940979003906

Test accuracy: 0.6913999915122986


RMSPROP:

Epoch 1/3: 938/938 [==============================] - 112s 118ms/step - loss: 0.3738 - accuracy: 0.8764

Epoch 2/3: 938/938 [==============================] - 109s 116ms/step - loss: 0.1064 - accuracy: 0.9661

Epoch 3/3: 938/938 [==============================] - 109s 116ms/step - loss: 0.0722 - accuracy: 0.9771

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

Epoch 1/3: 118/118 [==============================] - 130s 1s/step - loss: 0.4360 - accuracy: 0.8619

Epoch 2/3: 118/118 [==============================] - 130s 1s/step - loss: 0.2844 - accuracy: 0.9108

Epoch 3/3: 118/118 [==============================] - 130s 1s/step - loss: 0.2259 - accuracy: 0.9283

313/313 [==============================] - 9s 29ms/step - loss: 34.1011 - accuracy: 0.9152

Test loss: 34.10105895996094

Test accuracy: 0.9151999950408936

The performance of this model is good but worse than regular CNN, as seen from the accuracy and loss values.

# Hourglass CNN:

The code is available in hourglass_CNN.ipynb notebook file.

With Adam optimizer, epochs = 3, and batch size = 512, below loss and accuracy have been achieved:

Epoch 1/3: 118/118 [==============================] - 86s 721ms/step - loss: 2.3020 - accuracy: 0.1115

Epoch 2/3: 118/118 [==============================] - 85s 721ms/step - loss: 2.3014 - accuracy: 0.1124

Epoch 3/3: 118/118 [==============================] - 85s 718ms/step - loss: 2.3012 - accuracy: 0.1124

313/313 [==============================] - 11s 33ms/step - loss: 73.6535 - accuracy: 0.1033

Test loss: 73.65352630615234

Test accuracy: 0.10329999774694443

This model performance is worst among the three CNN models. It’s accurate in only 10% cases for both training and test data.

As a summary, to have an optimized performance, we will go ahead with regular CNN with RMSProp optimizer, batch size = 64, epochs = 3 (more can be better but computational time will increase significantly with little amount of improvement in accuracy) and learning rate 0.001.

#LeNet CNN:

A LeNet CNN has been coded and evaluated using cifar-10 dataset in keras. The code can be found in lenet_CNN.ipynb colab notebook file. 

The results with adam optimizer, epochs = 3, batch size =64 are presented below:

Epoch 1/3: 782/782 [==============================] - 90s 114ms/step - loss: 1.5481 - accuracy: 0.4391

Epoch 2/3: 782/782 [==============================] - 90s 115ms/step - loss: 1.2125 - accuracy: 0.5704

Epoch 3/3: 782/782 [==============================] - 89s 114ms/step - loss: 1.0430 - accuracy: 0.6329

313/313 [==============================] - 8s 26ms/step - loss: 148.4995 - accuracy: 0.5212

Test loss: 148.49952697753906

Test accuracy: 0.5212000012397766


- To study the effects of learning rate on the model performance, we varied learning rate in Adam optimizer with batch size 64 and epochs =1. The accuracy results are presented in below table

![alt text](https://github.com/Zobaer/ConvolutionalNeuralNet/blob/main/figs/Table4.png)

It is noticed that the accuracies are not that great but .001 gives the best accuracy. To improve accuracy further, we have to increase the number of epochs with learning rate 0.001.

- To study the effects of batch size on the model performance, we varied batch size with Adam optimizer, learning rate = 0.001 and epochs =1. The accuracy results are presented in below table:

![alt text](https://github.com/Zobaer/ConvolutionalNeuralNet/blob/main/figs/Table5.png)


It is hard to determine the pattern from the above table but batch size 16 or 512 should work better than others, based on the achieved training and test accuracies.

- To determine the best hyperparameters, we will use the knowledge gained from (1) and (2) and will run the training process for higher number of epochs. 
Epochs = 5, learning rate = 0.001 and batch size = 512 give below results:

Epoch 1/5: 98/98 [==============================] - 78s 790ms/step - loss: 1.8049 - accuracy: 0.3464

Epoch 2/5: 98/98 [==============================] - 77s 790ms/step - loss: 1.4538 - accuracy: 0.4795

Epoch 3/5: 98/98 [==============================] - 77s 787ms/step - loss: 1.3110 - accuracy: 0.5331

Epoch 4/5: 98/98 [==============================] - 78s 791ms/step - loss: 1.2125 - accuracy: 0.5710

Epoch 5/5: 98/98 [==============================] - 78s 794ms/step - loss: 1.1451 - accuracy: 0.5987

313/313 [==============================] - 8s 26ms/step - loss: 216.8900 - accuracy: 0.4644

Test loss: 216.88995361328125

Test accuracy: 0.4643999934196472


Epochs = 5, learning rate = 0.001 and batch size = 16 give below results:

Epoch 1/5: 3125/3125 [==============================] - 77s 24ms/step - loss: 1.4834 - accuracy: 0.4629

Epoch 2/5: 3125/3125 [==============================] - 78s 25ms/step - loss: 1.1116 - accuracy: 0.6050

Epoch 3/5: 3125/3125 [==============================] - 78s 25ms/step - loss: 0.9295 - accuracy: 0.6732

Epoch 4/5: 3125/3125 [==============================] - 77s 25ms/step - loss: 0.7896 - accuracy: 0.7222

Epoch 5/5: 3125/3125 [==============================] - 77s 25ms/step - loss: 0.6681 - accuracy: 0.7652


313/313 [==============================] - 5s 17ms/step - loss: 199.1083 - accuracy: 0.5234

Test loss: 199.1082763671875

Test accuracy: 0.5234000086784363

It is clear that batch size 16 is working best but more number of epochs is needed to improve test accuracy. Another training with Epochs = 25, learning rate = 0.001 and batch size = 16 give below results:

Epoch 1/25: 3125/3125 [==============================] - 114s 36ms/step - loss: 1.4793 - accuracy: 0.4651

Epoch 2/25: 3125/3125 [==============================] - 113s 36ms/step - loss: 1.1245 - accuracy: 0.6028

Epoch 3/25: 3125/3125 [==============================] - 113s 36ms/step - loss: 0.9603 - accuracy: 0.6627

Epoch 4/25: 3125/3125 [==============================] - 114s 36ms/step - loss: 0.8284 - accuracy: 0.7095

Epoch 5/25: 3125/3125 [==============================] - 113s 36ms/step - loss: 0.7051 - accuracy: 0.7521

Epoch 6/25: 3125/3125 [==============================] - 111s 36ms/step - loss: 0.5844 - accuracy: 0.7939

Epoch 7/25: 3125/3125 [==============================] - 111s 36ms/step - loss: 0.4765 - accuracy: 0.8315

Epoch 8/25: 3125/3125 [==============================] - 111s 36ms/step - loss: 0.3887 - accuracy: 0.8618

Epoch 9/25: 3125/3125 [==============================] - 111s 36ms/step - loss: 0.3192 - accuracy: 0.8865

Epoch 10/25: 3125/3125 [==============================] - 111s 36ms/step - loss: 0.2699 - accuracy: 0.9036

Epoch 11/25: 3125/3125 [==============================] - 112s 36ms/step - loss: 0.2388 - accuracy: 0.9175

Epoch 12/25: 3125/3125 [==============================] - 112s 36ms/step - loss: 0.2070 - accuracy: 0.9286

Epoch 13/25: 3125/3125 [==============================] - 112s 36ms/step - loss: 0.1964 - accuracy: 0.9331

Epoch 14/25: 3125/3125 [==============================] - 112s 36ms/step - loss: 0.1789 - accuracy: 0.9394

Epoch 15/25: 3125/3125 [==============================] - 112s 36ms/step - loss: 0.1707 - accuracy: 0.9426

Epoch 16/25: 3125/3125 [==============================] - 112s 36ms/step - loss: 0.1510 - accuracy: 0.9481

Epoch 17/25: 3125/3125 [==============================] - 113s 36ms/step - loss: 0.1525 - accuracy: 0.9497

Epoch 18/25: 3125/3125 [==============================] - 113s 36ms/step - loss: 0.1443 - accuracy: 0.9535

Epoch 19/25: 3125/3125 [==============================] - 114s 37ms/step - loss: 0.1466 - accuracy: 0.9535

Epoch 20/25: 3125/3125 [==============================] - 112s 36ms/step - loss: 0.1324 - accuracy: 0.9576

Epoch 21/25: 3125/3125 [==============================] - 111s 36ms/step - loss: 0.1296 - accuracy: 0.9587

Epoch 22/25: 3125/3125 [==============================] - 111s 36ms/step - loss: 0.1252 - accuracy: 0.9588

Epoch 23/25: 3125/3125 [==============================] - 111s 36ms/step - loss: 0.1237 - accuracy: 0.9612

Epoch 24/25: 3125/3125 [==============================] - 111s 36ms/step - loss: 0.1300 - accuracy: 0.9599

Epoch 25/25: 3125/3125 [==============================] - 111s 36ms/step - loss: 0.1150 - accuracy: 0.9637

313/313 [==============================] - 8s 26ms/step - loss: 751.0416 - accuracy: 0.5269

Test loss: 751.0416259765625

Test accuracy: 0.5268999934196472

The accuracy kept improving with increasing number of epochs. It’s worth plotting the accuracy and loss data against number of epochs. The plots look like below (from tensorboard):

![alt text](https://github.com/Zobaer/ConvolutionalNeuralNet/blob/main/figs/Graph1.png)
Fig. 1: Accuracy data plotted against number of epochs (learning rate = 0.001, batch size = 16, algorithm used: LeNet CNN)

![alt text](https://github.com/Zobaer/ConvolutionalNeuralNet/blob/main/figs/Graph2.png)
Fig. 2: Loss data plotted against number of epochs (learning rate = 0.001, batch size = 16, algorithm used: LeNet CNN)

- The number of parameters in CNN_lenet can be seen from below output:

![alt text](https://github.com/Zobaer/ConvolutionalNeuralNet/blob/main/figs/Table5.png)

# Equivalent Feedforward Network

The equivalent feedforward network has been implemented. The code is available in LeNet_equivalent_feedforward_network.ipynb file. With feedforward network, the number of parameters became extremely high as seen from below result:

![alt text](https://github.com/Zobaer/ConvolutionalNeuralNet/blob/main/figs/Table6.png)

- Due to every possible cross-connections in a fully connected layers, we got very high number of parameters in the equivalent dense network. Thus, model became computationally less efficient, each epoch was taking around 47 minutes thus running 25 epochs was nearly impossible. 
- The number of parameters in CNN_lenet was 697,046 which was well manageable even for running 25 epochs to get high accuracy. But the equivalent feedforward network had 122,854,454 parameters which is computationally inefficient to a great extent, convergence was really slower and it could overfit the model too. Hence, feedforward network is not preferred in such this classification task.







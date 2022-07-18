## cnn-pytorch-experiments

Taco Cohen's experiments are being reproduced in this repository with the aim of continuing this line of study and improving equivariance and invariance in artificial neural networks.

### Results

Here you can see the results of evaluating the different models with the test data set rotated from -180° to 180°.

| Model used           | Plot results                                                              | Details                                                                                                                                                                                                   |
|----------------------|---------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Z2CNN                | <img src="./evaluation/eval360Z2CNN.png" height="auto" width="300">       | The expected behavior is obtained since the layers of this model do not have equivariance, so it is normal that the maximum number of hits is around 0.                                                   |
| Z2CNN (ROT-MNIST)    | <img src="./evaluation/eval360Z2CNNROT.png" height="auto" width="300">    | The oiling rate has improved considerably with respect to normal MNIST at angles far from 0.                                                                                                              |
| P4CNNP4              | <img src="./evaluation/eval360P4CNNP4.png" height="auto" width="300">     | As this model has equivariance layers, we expected to obtain an improvement in the invariance to rotations and to achieve a better hit rate at angles far from 0º. It is very similar to the Z2CNN model. |
| P4CNNP4 (ROT-MNIST)  | <img src="./evaluation/eval360P4CNNP4ROT.png" height="auto" width="300">  | In this case, it has increased the hit rate from 91% to 94%. A much more stable curve is observed.                                                                                                        |
| ConvEq2D             | <img src="./evaluation/eval360ConvEq2D.png" height="auto" width="300">    | -                                                                                                                                                                                                         |
| ConvEq2D (ROT-MNIST) | <img src="./evaluation/eval360ConvEq2DROT.png" height="auto" width="300"> | -                                                                                                                                                                                                         |

    

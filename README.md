<!-- ABOUT THE PROJECT -->
## cnn-pytorch-experiments

Taco Cohen's experiments are being reproduced in this repository with the aim of continuing this line of study and improving equivariance and invariance in artificial neural networks.

### Results

Here you can see the results of evaluating the different models with the test data set rotated from -180° to 180°.

| Model used          | Plot results                                                           | Details                                                                                                                     |
|---------------------|------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| Z2CNN               | <img src="./evaluation/eval360Z2CNN.png" height="150" width="auto">    |                                                                                                                             |
| Z2CNN (ROT-MNIST)   | <img src="./evaluation/eval360Z2CNNROT.png" height="150" width="auto"> |                                                                                                                             |
| P4CNNP4             | <img src="./evaluation/eval360P4CNNP4.png" height="150" width="auto">  | It can be seen that the rotation invariance is not working properly since at angles far from 0º it has a very low hit rate. |
| P4CNNP4 (ROT-MNIST) | <img src="" height="150" width="auto">                                 |                                                                                                                             |

    

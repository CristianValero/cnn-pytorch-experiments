<!-- ABOUT THE PROJECT -->
## cnn-pytorch-experiments

Taco Cohen's experiments are being reproduced in this repository with the aim of continuing this line of study and improving equivariance and invariance in artificial neural networks.

### Results

* P4CNNP4: This is the result of evaluating the P4CNNP4 model with the dataset rotated from -180º to 180º. It can be seen that the rotation invariance is not working properly 
since at angles far from 0º it has a very low hit rate.

    

| Model used | Plot results                                             |
|------------|----------------------------------------------------------|
| Z2CNN      | <img src="./evaluation/eval360P4CNNP4.png" height="300"> |
| P4CNNP4    | <img src="./evaluation/eval360P4CNNP4.png" height="300"> |
    

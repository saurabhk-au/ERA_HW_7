# ERA_HW7_CNN
CNN 
     # MNIST Assignment

     This repository contains the implementation of a neural network model for the MNIST dataset.

     ## Model Details

     - **Total Parameter Count:** 7662 parameters
     - **Batch Normalization:** Yes
     - **DropOut:** Yes
     - **Fully Connected Layer or GAP:** NO
     - GAP yes
     - LR Scheduler : Yes
     - Rotation: Yes
     - Test Accuracy (best) 99.14%



Model Summary for Net:
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 10, 26, 26]              90
              ReLU-2           [-1, 10, 26, 26]               0
       BatchNorm2d-3           [-1, 10, 26, 26]              20
           Dropout-4           [-1, 10, 26, 26]               0
            Conv2d-5           [-1, 20, 24, 24]           1,800
              ReLU-6           [-1, 20, 24, 24]               0
       BatchNorm2d-7           [-1, 20, 24, 24]              40
           Dropout-8           [-1, 20, 24, 24]               0
            Conv2d-9           [-1, 10, 24, 24]             200
        MaxPool2d-10           [-1, 10, 12, 12]               0
           Conv2d-11           [-1, 10, 10, 10]             900
             ReLU-12           [-1, 10, 10, 10]               0
      BatchNorm2d-13           [-1, 10, 10, 10]              20
          Dropout-14           [-1, 10, 10, 10]               0
           Conv2d-15             [-1, 16, 8, 8]           1,440
             ReLU-16             [-1, 16, 8, 8]               0
      BatchNorm2d-17             [-1, 16, 8, 8]              32
          Dropout-18             [-1, 16, 8, 8]               0
           Conv2d-19             [-1, 20, 6, 6]           2,880
             ReLU-20             [-1, 20, 6, 6]               0
      BatchNorm2d-21             [-1, 20, 6, 6]              40
          Dropout-22             [-1, 20, 6, 6]               0
AdaptiveAvgPool2d-23             [-1, 20, 1, 1]               0
           Conv2d-24             [-1, 10, 1, 1]             200
================================================================
Total params: 7,662
Trainable params: 7,662
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.70
Params size (MB): 0.03
Estimated Total Size (MB): 0.73
----------------------------------------------------------------
EPOCH: 0: 100%|████████████████████████████████| 469/469 [01:13<00:00,  6.35it/s, Accuracy=86.57, Batch_id=468, loss=0.185]

Test set: Average loss: 0.1536, Accuracy: 9680/10000 (96.80%)

EPOCH: 1: 100%|███████████████████████████████| 469/469 [01:18<00:00,  5.99it/s, Accuracy=96.80, Batch_id=468, loss=0.0627]

Test set: Average loss: 0.0812, Accuracy: 9780/10000 (97.80%)

EPOCH: 2: 100%|███████████████████████████████| 469/469 [01:17<00:00,  6.03it/s, Accuracy=97.53, Batch_id=468, loss=0.0447]

Test set: Average loss: 0.0619, Accuracy: 9822/10000 (98.22%)

EPOCH: 3: 100%|███████████████████████████████| 469/469 [01:20<00:00,  5.80it/s, Accuracy=97.92, Batch_id=468, loss=0.0571]

Test set: Average loss: 0.0430, Accuracy: 9886/10000 (98.86%)

EPOCH: 4: 100%|███████████████████████████████| 469/469 [01:53<00:00,  4.15it/s, Accuracy=98.20, Batch_id=468, loss=0.0298]

Test set: Average loss: 0.0373, Accuracy: 9890/10000 (98.90%)

EPOCH: 5: 100%|███████████████████████████████| 469/469 [01:20<00:00,  5.83it/s, Accuracy=98.38, Batch_id=468, loss=0.0348]

Test set: Average loss: 0.0454, Accuracy: 9849/10000 (98.49%)

EPOCH: 6: 100%|███████████████████████████████| 469/469 [01:07<00:00,  6.99it/s, Accuracy=98.64, Batch_id=468, loss=0.0329]

Test set: Average loss: 0.0316, Accuracy: 9907/10000 (99.07%)

EPOCH: 7: 100%|███████████████████████████████| 469/469 [01:22<00:00,  5.68it/s, Accuracy=98.71, Batch_id=468, loss=0.0715]

Test set: Average loss: 0.0304, Accuracy: 9904/10000 (99.04%)

EPOCH: 8: 100%|███████████████████████████████| 469/469 [01:21<00:00,  5.74it/s, Accuracy=98.70, Batch_id=468, loss=0.0645]

Test set: Average loss: 0.0303, Accuracy: 9904/10000 (99.04%)

EPOCH: 9: 100%|████████████████████████████████| 469/469 [01:15<00:00,  6.17it/s, Accuracy=98.81, Batch_id=468, loss=0.023]

Test set: Average loss: 0.0309, Accuracy: 9907/10000 (99.07%)

EPOCH: 10: 100%|███████████████████████████████| 469/469 [01:24<00:00,  5.54it/s, Accuracy=98.78, Batch_id=468, loss=0.056]

Test set: Average loss: 0.0298, Accuracy: 9910/10000 (99.10%)

EPOCH: 11: 100%|██████████████████████████████| 469/469 [01:22<00:00,  5.70it/s, Accuracy=98.77, Batch_id=468, loss=0.0585]

Test set: Average loss: 0.0293, Accuracy: 9910/10000 (99.10%)

EPOCH: 12: 100%|██████████████████████████████| 469/469 [01:20<00:00,  5.80it/s, Accuracy=98.83, Batch_id=468, loss=0.0166]

Test set: Average loss: 0.0294, Accuracy: 9910/10000 (99.10%)

EPOCH: 13: 100%|███████████████████████████████| 469/469 [01:14<00:00,  6.30it/s, Accuracy=98.84, Batch_id=468, loss=0.131]

Test set: Average loss: 0.0293, Accuracy: 9911/10000 (99.11%)

EPOCH: 14: 100%|██████████████████████████████| 469/469 [01:38<00:00,  4.76it/s, Accuracy=98.85, Batch_id=468, loss=0.0783]

Test set: Average loss: 0.0293, Accuracy: 9914/10000 (99.14%)
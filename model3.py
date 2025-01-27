import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a dropout value					
dropout_value = 0.05 # Example value, adjust as needed					

# Target: Define a simple CNN model for MNIST classification.
# Result: Achieved a test accuracy of XX% after training.
# Analysis: The model uses a series of convolutional layers followed by a global average pooling layer.

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
                            
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.Dropout(dropout_value)
        ) # output_size = 24

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        ) # output_size = 11

        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11


        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        ) # output_size = 9
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 7

        # OUTPUT BLOCK
#        self.convblock6 = nn.Sequential(
#           nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
#           nn.ReLU(),
#           nn.BatchNorm2d(16),
#        ) # output_size = 7        
        #nn.AdaptiveAvgPool((1,1))
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.Dropout(dropout_value)
        ) # output_size = 

        self.gap = nn.Sequential(
        #    nn.AvgPool2d(kernel_size=6)
            nn.AdaptiveAvgPool2d((1,1))
        ) # output_size = 1
    
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        ) # output_size = 

        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
  #     x = self.convblock7(x)
        x = self.gap(x)
        x = self.convblock9(x)
        

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
# The primary neural network that will serve as the backbone of this project will be a convolutional neural network (CNN). Such a network consists of an encoder which involves a 
# series of convolutions and pooling layers, both of which are used to consolidate information and extract important details; this part is also known as feature learning. 
# Through convolutions, the network applies weighted matrices (kernels) to extract significant details and patterns from the input data, whereas the pooling layers are used to 
# consolidate this information while maintaining the important features. Each convolution layer will consist of multiple 3x3 kernels, and each pooling layer will 
# consist of 2x2 kernels and will have a max-pooling effect.

# To introduce non-linearity and enable the network to analyze data in a higher-dimensional space, the Rectified Linear Unit (ReLU) activation function is employed 
# after each convolutional operation. The ReLU function is a simple function which transforms negative values to zero and keeps positive values unaltered.

# After the encoder, there is a classifier which flattens the multidimensional information to a one dimensional vector that is used 
# for predictions. Since feature learning is the more arduous part of this network, and occurs in the encoder, the classifier will likely have few (1-3) 
# layers, contrary to the encoder which will incorporate multiple layers. 

# In summary, we will incorporate a convolutional neural network, consisting of an encoder that is responsible for feature learning, while using 
# convolutions to identify crucial information and pooling to condense it. By incorporating the ReLU activation function, the network gains the capacity to 
# comprehend the data in a more intricate manner, permitting the classifier to more effectively isolate the different classes that exist, thus 
# allowing for more accurate predictions.


# Image size: 256 x 256
# HSV colour space

class FruitRipenessDetector(nn.Module):
    def __init__(self):
         super(FruitRipenessDetector, self).__init__()
         self.name = "ripeness_detector"
         self.conv1 = nn.Conv2d(3, 50, 3, 1, 1) # in_channels = 3 (HSV), out_channels = 50 , kernel_size = 3x3, stride = 1, padding = 1 (to preserve resolution)
         self.pool = nn.MaxPool2d(2, 2) # max pooling for feature learning, repreated after every iteration
         self.conv2 = nn.Conv2d(50, 100, 3, 1, 1) # in_channels = 50 (output of conv1), out_channels = 100, everything else remains the same, keep adding +50 layers
         self.conv3 = nn.Conv2d(100, 150, 3, 1, 1)
         self.conv4 = nn.Conv2d(150, 200, 3, 1, 1) 
         self.conv5 = nn.Conv2d(200, 250, 3, 1, 1)
         self.conv6 = nn.Conv2d(250, 300, 3, 1, 1)
         self.fc1 = nn.Linear(300 * 8 * 8 , 32) # 300 * 8 * 8 output channels after conv6, reduced to 32 output features (arbitrary)
         self.fc2 = nn.Linear(32, 1) # 32 features reduced to 1 dimension for binary classification

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = self.pool(F.relu(self.conv6(x)))
        x = x.view(-1, 300 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.squeeze(1) # Flatten to [batch_size]
        return x

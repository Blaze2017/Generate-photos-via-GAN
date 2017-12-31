# Deep Convolutional GANs

# Importing the libraries
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

# Setting some hyperparameters
batchSize = 64 # We set the size of the batch.
imageSize = 64 # We set the size of the generated images (64x64).

# Creating the transformations
transform = transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]) # We create a list of transformations (scaling, tensor conversion, normalization) to apply to the input images.

# Loading the dataset
dataset = dset.CIFAR10(root = './data', download = True, transform = transform) # We download the training set in the ./data folder and we apply the previous transformations on each image.
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers = 2) # We use dataLoader to get the images of the training set batch by batch.

# Defining the weights_init function that takes as input a neural network m and that will initialize all its weights.
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Defining the generator

class G(nn.Module): # We introduce a class to define the generator.
    
    # we inherit from the abstract class -- nn.Module
    

    def __init__(self): # We introduce the __init__() function that will define the architecture of the generator.
        super(G, self).__init__() # super function to initilize child class
        self.main = nn.Sequential( # We create a meta module of a neural network that will contain a sequence of modules (convolutions, full connections, etc.).
            
            # Add the 1st layer of convolution    
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias = False), # We start with an inversed convolution.
            
            # 100: input map features
            # 512: output map features
            # 4: Kernel size
            # 1: Stride size
            # 0: padding
            
            nn.BatchNorm2d(512), 
            nn.ReLU(True), #Activation function ReLU
            
            # Add the 2nd layer of convolution
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False), # We start with an inversed convolution.
            
            # The output of the last layer is 512.
            # Thus, we need 512 inputs now
            
            
            nn.BatchNorm2d(256), 
            nn.ReLU(True), #Activation function ReLU
            
            
            # Add the 3rd layer of convolution
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False), # We start with an inversed convolution.
            nn.BatchNorm2d(128), 
            nn.ReLU(True), #Activation function ReLU

            # Add the 4th layer of convolution
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False), # We start with an inversed convolution.
            nn.BatchNorm2d(64), 
            nn.ReLU(True), #Activation function ReLU

            # Add the 5th layer of convolution
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False), # We start with an inversed convolution.
            
            # As there are 3 color channels for the images,
            # we need to output 3 nodes
            
            
        )
        
    def forward(self, input):
        output = self.main(input) #Get the output of the generator
        return output
    
# Creating the generator
netG = G() # we use the previously defined class G to generate a class netG
netG.apply(weights_init) # Apply weigh initilization function to our neural network

# Define the discriminator class
class D(nn.Module): # We use nn.Module again as our parent class
    def __init__(self):
        super(D, self).__init__()
        self.main = nn.Sequential(
                
                # First layer
                nn.Conv2d(3, 64, 4, 2, 1, bias = False), # First module
                
                # As there are 3 outputs in the generator,
                # we need to take these 3, and then out put 64 map features
                
                nn.LeakyReLU(0.2, inplace = True), # Second module
                
                # LeakReLu has a negative value in the range when x <  0. 
                # Thus, we can choose a slope for this segment, e.g., 0.2
                
                
                
                # Second layer
                nn. Conv2d(64, 128, 4, 2, 1, bias = False), 
                nn. BatchNorm2d(128), # Normaliz the output feature maps
                nn.LeakyReLU(0.2, inplace = True), # Second module
                
                # Third layer
                nn. Conv2d(128, 256, 4, 2, 1, bias = False), 
                nn. BatchNorm2d(256), # Normaliz the output feature maps
                nn.LeakyReLU(0.2, inplace = True), # Second module
                
                # Forth layer
                nn. Conv2d(256, 512, 4, 2, 1, bias = False), 
                nn. BatchNorm2d(512), # Normaliz the output feature maps
                nn.LeakyReLU(0.2, inplace = True), # Second module
                
                # Fifth layer
                nn. Conv2d(512, 1, 4, 1, 0, bias = False), 
                nn.Sigmoid()
                )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1) # Flattern the output
    
# Creating the discriminator
netD = D()
netD.apply(weights_init)

# Training the DCGANs
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr = 0.0002, betas = (0.5, 0.999)) 
optimizerG = optim.Adam(netG.parameters(), lr = 0.0002, betas = (0.5, 0.999))
       
for epoch in range(25): 
    for i, data in enumerate(dataloader, 0):
        # 1st step: Updating the weights of the neural network of the discriminator
        netD.zero_grad()
        
        # Training the discriminator with the 
        real, _ = data
        input = Variable(real)
        target = Variable(torch.ones(input.size()[0])) # unit matrix
        # True result: all should be 1
        output = netD(input)
        errD_real = criterion(output, target)
        
        # Training the dsicriminator with fake image generated by the generator
        noise = Variable(torch.randn(input.size()[0], 100, 1, 1))
        # get 100 feature maps with size 1 by 1
        fake = netG(noise)
        target = Variable(torch.zeros(input.size()[0]))
        output = netD(fake.detach())
        errD_fake = criterion(output, target)
        
        # Backpropagating the total error
        errD = errD_real + errD_fake
        errD.backward()
        optimizerD.step()
        
        
        # 2nd Step: Updating the weights of the neural network of the generators
        netG.zero_grad()
        target = Variable(torch.ones(input.size()[0]))
        output = netD(fake)
        errG = criterion(output, target)
        errG.backward()
        optimizerG.step()
        
        # 3rd step: Prinssting the losses and saving the real images and the generated images
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, 25, i, len(dataloader), errD.data[0], errG.data[0]))
        if i % 100 == 0:
            vutils.save_image(real, '%s/real_samples.png' % "./results", normalize = True) # We save the real images of the minibatch.
            fake = netG(noise) # We get our fake generated images.
            vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ("./results", epoch), normalize = True) # We also save the fake generated images of the minibatch.
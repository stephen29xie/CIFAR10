# CIFAR10

This is a project I did to learn about and implement Convolutional Neural Networks for image classification. I tackled the CIFAR-10 dataset, a dataset with 60,000 32x32 images (50000 training, 10000 testing), each belonging to one of 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).
More information here: https://www.cs.toronto.edu/~kriz/cifar.html

I use Pytorch as my deep learning framework. My goal was to architect and train a competitive ConvNet from scratch. After experimenting with many architectures and tuning parameters, my final solution is an 11 layer ConvNet which achieved 91.94% testing accuracy.

Please see the jupyter notebook for the full implementation, from data prep to training and testing. Training the network took me ~1 hour on a NVIDIA V100 GPU instance using Google Compute Engine services.
%Create a CNN with 4 classification outputs
%Image Data Augmentation
%Checkpoint Saving
%Maybe use a Deeper Network?
%Can Overfitting and Underfitting be avoided?
%Dropout for Overfitting or Early Stopping method or Cross Validation ?.
%Deep Layers for Underfitting? Fine tune parameters.
clear all;
close all;
clc;
    

%Set the path to the Training and Validation Dataset
dcbhDatasetPath = fullfile('./DogCatHorseBear');
dcbhDatasetPath1 = fullfile('./Testing');

%Checkpoint
checkpointPath = 'C:\Users\Rushabh\Documents\MATLAB\Deep Learning\Assignment 3';
%ImageDatastore Stores and Labels all the data into a .tiff file.
imds = imageDatastore(dcbhDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
testImg = imageDatastore(dcbhDatasetPath1, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
%Label Count
labelCount = countEachLabel(imds);
%Network Architecture
layers = [
    imageInputLayer([256 256 3])
   
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
   
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer
    dropoutLayer
    fullyConnectedLayer(4)
    softmaxLayer
    classificationLayer];

%Define the Input Size
inputSize = [256 256 3];

%Split the data into Training and Validation Set. Here: 75% and 25% resp.
%numTrainFiles = 1026;
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.75,'randomize');

%Transformation Matrix for augmentation
pixelRange = [-130 130];


%Options defined, random translations and rotations on the training images.
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandScale',[2 2], ...
    'RandXShear',[2 2], ...
    'RandRotation',[-180 180]);

%Augment the training data.
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter,'ColorPreprocessing','gray2rgb');

%The validation images should also be as the same size as the Training so
%just resized the data. No augmentation done. Except for color correction.
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation,'ColorPreprocessing','gray2rgb');
testing = augmentedImageDatastore(inputSize(1:2),testImg,'ColorPreprocessing','gray2rgb');

%Training options: Can change Epochs, learning rate, solver and
%L2Regularization for fine tuning/increasing accuracy.
options = trainingOptions('adam', ...
    'InitialLearnRate',1e-4, ...
    'MaxEpochs',1, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',2, ...
    'Verbose',false, ...
    'L2Regularization',0.005, ...
    'Plots','training-progress', ...
    'CheckpointPath',checkpointPath);

load('net_checkpoint__95__2019_06_26__18_23_56.mat','net');

%net1 = trainNetwork(augimdsTrain,net.Layers,options);
%net = trainNetwork(augimdsTrain,layers,options);%Train the network on Augmented Images

%Calculate the accuracy.
[YPred,probs] = classify(net,testing);
accuracy = mean(YPred == testImg.Labels);
figure
cm = confusionchart(testImg.Labels,YPred);
disp(accuracy);
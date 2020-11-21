%TRANSFERLEARNING1
% 
% Author: https://github.com/juancarlosmiranda/
% Date: November 2020
%
% Based on Mathworks - Deep Learning ONRAMP
% 
% Create a pre-trained network
% Save this network on a file
%    
% 
% USAGE:
% >> help TranferLearning1
% 
% HELP:
% >> help TranferLearning1
%
% Load pre-trained Alexnet
% Transfer learning with flowers database
% Train Googlenet with flowers database
% Save in file flowersnet.m

% ---------------------
imds=imageDatastore(ConfigData.pathFlowersDataset, 'IncludeSubfolders',true,'LabelSource','foldernames');
%montage(imds); % is to view images in imageDatastore

[trainImgs,testImgs] = splitEachLabel(imds, ConfigData.trainSplitValue);
% image pre-processing, adapt to 227*227 pixels
trainds = augmentedImageDatastore(ConfigData.pixelSizeNeuralNetowrk, trainImgs);
testds = augmentedImageDatastore(ConfigData.pixelSizeNeuralNetowrk, testImgs);

% define network AlexNet and train
myNet = alexnet;
layers = myNet.Layers;
layers(end-2) = fullyConnectedLayer(ConfigData.numberOfLabelsDataset); % change categories to a new dataset
layers(end) = classificationLayer(); % replace last layer to classify the new dataset
options = trainingOptions(ConfigData.optionTraining,'InitialLearnRate',ConfigData.learnRate);

% train network with some options
myNet = trainNetwork(trainds,layers,options);

% save network in a file to use in other algorithm
save(ConfigData.pathSaveFlowerNet, 'myNet');


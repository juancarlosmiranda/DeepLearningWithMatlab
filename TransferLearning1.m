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
% Load pre-trained Googlenet
% Transfer learning with flowers database
% Train Googlenet with flowers database
% Save in file flowersnet.m

% ------------------------------------------------------------------------
% configuration parameters
% ------------------------------------------------------------------------
HOMEUSER=strcat(pwd,'/');
mainPath = fullfile(HOMEUSER,'development');

% configuration for dataset
pathFlowersDataset = fullfile(mainPath,'datasets_deep_learning','Flowers');
numberOfLabelsDataset = 12;


% config parameters for model
neuralNetName = 'flowerNet2.mat';
pathSaveFlowerNet = fullfile(mainPath,'trained_models', neuralNetName);


trainSplitValue = 0.8 % split datastore in 80 percent
pixelSizeNeuralNetowrk = [227 227];
learnRate=0.0001;
optionTraining='sgdm';

% ---------------------
imds=imageDatastore(pathFlowersDataset, 'IncludeSubfolders',true,'LabelSource','foldernames');
%montage(imds); % is to view images in imageDatastore

[trainImgs,testImgs] = splitEachLabel(imds, trainSplitValue);
% image pre-processing, adapt to 227*227 piels
trainds = augmentedImageDatastore(pixelSizeNeuralNetowrk, trainImgs);
testds = augmentedImageDatastore(pixelSizeNeuralNetowrk, testImgs);

% define network AlexNet and train
myNet = alexnet;
layers = myNet.Layers;
layers(end-2) = fullyConnectedLayer(numberOfLabelsDataset); % change categories to a new dataset
layers(end) = classificationLayer(); % replace last layer to classify the new dataset
options = trainingOptions(optionTraining,'InitialLearnRate',learnRate);

% train network with some options
myNet = trainNetwork(trainds,layers,options);

% save network in a file to use in other algorithm
save(pathSaveFlowerNet, 'myNet');


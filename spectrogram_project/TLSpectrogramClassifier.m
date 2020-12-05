%TLSPECTROGRAMCLASSIFIER
% 
% Author: https://github.com/juancarlosmiranda/
% Date: November 2020
%
% Code taken from the Deep Learning with Matlab course.
% 
% With a database of spectrograms images, this script train a neural
% network Alexnet, make transfer learning to classify 14 musical
% instruments.
%
% This is assumed to be histogram image inputs of different sounds, stored in a folder.
% Show a confusion matrix and accuracy
%    
% 
% USAGE:
% >> help TLSpectrogram
% 
% HELP:
% >> help TLSpectrogram
%
% Load pre-trained Alexnet
% Transfer learning with spectrograms database
% Train Alexnet with spectrograms database
% Save in file SoundNeuralNet.mat
% ---------------------
% use demo dataset from Matlab repository
imds=imageDatastore(ConfigData.pathDataset, 'IncludeSubfolders',true,'LabelSource','foldernames');
labelCount = countEachLabel(imds)

% Specify Training and Validations Sets
[imdsTrain,imdsTest, imdsValidation]= splitEachLabel(imds, ConfigData.trainSplitValue, 0.1, 0.1, 'randomize');


% image pre-processing, adapt to 227*227 pixels
trainds = augmentedImageDatastore(ConfigData.pixelSizeNeuralNetowrk, imdsTrain);
testds = augmentedImageDatastore(ConfigData.pixelSizeNeuralNetowrk, imdsTest);
validationds = augmentedImageDatastore(ConfigData.pixelSizeNeuralNetowrk, imdsTest);


% Specify Training options
%options = trainingOptions('sgdm','InitialLearnRate',0.0001,'MaxEpochs',2,'Plots','training-progress');
options = trainingOptions(ConfigData.optionTraining,'InitialLearnRate',ConfigData.learnRate, 'MaxEpochs', ConfigData.maxEpochs, 'Shuffle','every-epoch', 'ValidationData',validationds, 'ValidationFrequency',ConfigData.ValidationFrequency, 'Verbose',false, 'Plots','training-progress');


% Train Network, define network AlexNet and train
SpectrogramNeuralNet = alexnet;
inputSize = SpectrogramNeuralNet.Layers(1).InputSize

% Get layers from neural network
layers = SpectrogramNeuralNet.Layers;
layers(end-2) = fullyConnectedLayer(ConfigData.numberOfLabelsDataset); % change categories to a new dataset
layers(end) = classificationLayer(); % replace last layer to classify the new dataset

% Train network with some options
SpectrogramNeuralNet = trainNetwork(trainds,layers,options);

% Save network in a file to use in other algorithm
save(ConfigData.pathSaveNeuralNet, 'SpectrogramNeuralNet');

% Taken from Matlab examples
% Calculates Accuracy
testPred = classify(SpectrogramNeuralNet,testds);
nnz(testPred == imdsTest.Labels)/numel(imdsTest.Labels)

%Plot confusion matrix
[cmap,clabel] = confusionmat(imdsTest.Labels,testPred);
heatmap(clabel,clabel,cmap)

classdef ConfigData
   properties (Constant)
      % ---------------------------------------------
      mainPath = fullfile('/','home','usuario','development'); % here configure your path
      % ---------------------------------------------
      % configuration for dataset, here configure your path
      pathDataset = fullfile(ConfigData.mainPath,'datasets_deep_learning','Spectrograms');
      numberOfLabelsDataset = 14;
      trainSplitValue = 0.8 % split datastore in 80 percent      
      % ---------------------------------------------
      % config parameters for model
      pixelSizeNeuralNetowrk = [227 227];
      learnRate=0.001;
      optionTraining='sgdm';
      maxEpochs = 40;
      ValidationFrequency = 50;
      neuralNetName = 'SoundNeuralNet.mat';      
      pathSaveNeuralNet = fullfile(ConfigData.mainPath,'trained_models', ConfigData.neuralNetName);            
      % ---------------------------------------------
   end   
end

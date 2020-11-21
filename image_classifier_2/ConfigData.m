classdef ConfigData
   properties (Constant)
      % ---------------------------------------------
      mainPath = fullfile('/','home','usuario','development');
      % ---------------------------------------------
      % configuration for dataset
      pathFlowersDataset = fullfile(ConfigData.mainPath,'datasets_deep_learning','Flowers');
      numberOfLabelsDataset = 12;
      trainSplitValue = 0.8 % split datastore in 80 percent      
      % ---------------------------------------------
      % config parameters for model
      pixelSizeNeuralNetowrk = [227 227];
      learnRate=0.0001;
      optionTraining='sgdm';            
      neuralNetName = 'flowerNet2.mat';      
      pathSaveFlowerNet = fullfile(ConfigData.mainPath,'trained_models', ConfigData.neuralNetName);            
      % ---------------------------------------------
   end   
end
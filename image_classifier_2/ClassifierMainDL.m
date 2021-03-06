classdef ClassifierMainDL   
    % CLASSIFIERMAINDL
    % Author: https://github.com/juancarlosmiranda/
    % 
    % 
    % Based on Mathworks Tutorial "Classify Webcam Images Using Deep Learning"
    % https://es.mathworks.com/help/deeplearning/ug/classify-images-from-webcam-using-deep-learning.html?s_tid=srchtitle
    % 
    % I adapted this to OOP format, is a practice to develop code for
    % Deep Learning. It uses a pre-trained Alexnet to classify objects.
    %
    % USAGE:
    % Run it with ->
    % ClassifierMainDL.Main()

    
    properties

    end    
    % ----------------------
    methods(Static)               
        function Main()
            clc, clear all, close all;
            fprintf('\n -------------------------------- \n');
            fprintf('MAIN METHOD - demo transfer learning');
            fprintf('\n -------------------------------- \n');
            
            % configure webcam, by default is 1
            % neural network by default alexnet
            fprintf('\n loading flower net, transfer learning \n');
            load (ConfigData.pathSaveFlowerNet) % neural net trained with flowers
            classifier_real_time = DLClassifier(myNet);
            classifier_real_time.runner;
        end        
    end
end

    

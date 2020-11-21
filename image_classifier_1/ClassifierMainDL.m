classdef ClassifierMainDL   
    % Author: https://github.com/juancarlosmiranda/
    % 
    %ClassifierMainDL
    % 
    % Based on Mathworks Tutorial "Classify Webcam Images Using Deep Learning"
    % https://es.mathworks.com/help/deeplearning/ug/classify-images-from-webcam-using-deep-learning.html?s_tid=srchtitle
    % 
    % I adapted this to OOP format, is a practice to develop code for
    % Deep Learning. It uses a pre-trained Alexnet to classify objects.
    % 
    % Run it with ->
    % c = DeepLearningClassifier;
    % c.runner()    
    
    properties
        Property1;
    end    
    % ----------------------
    methods(Static)
        function Main()
            clc, clear all, close all;
            fprintf('\n -------------------------------- \n');
            fprintf('MAIN METHOD');
            fprintf('\n -------------------------------- \n');
            classifier_real_time = DeepLearningClassifier;
            classifier_real_time.runner;
        end        
    end
end

    

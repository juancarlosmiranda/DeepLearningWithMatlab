classdef DLClassifier
    % Author: https://github.com/juancarlosmiranda/
    % 
    %   Here this is an detailed explainded text
    % Based on Mathworks Tutorial "Classify Webcam Images Using Deep Learning"
    % https://es.mathworks.com/help/deeplearning/ug/classify-images-from-webcam-using-deep-learning.html?s_tid=srchtitle
    % 
    % I adapted this to OOP format, is a practice to develop code for
    % Deep Learning. It uses a pre-trained Alexnet to classify objects.
    %
    % This improve load a neural network with transfer learning data to 
    % detect flowers.
    % 
    % Run it with ->
    % c = DLClassifier;
    % c.runner()

    
    % TODO: 
    % Add load of neural network trained from file
    % clean parameters
    
    properties
        camera
        nnet
        picture_to_classify
    end
    
    methods
        function obj = DLClassifier(nnet_in)
            %DeepLearningClassifier Construct instance
            fprintf('\n Constructor-> DLClassifier()');
            if nargin < 1
                fprintf('\n defaul Alexnet \n');
                % get AlexNet pre-trained
                obj.nnet = alexnet;
            else
                fprintf('\n loading nnet \n');                
                obj.nnet = nnet_in;
            end
        end        
        
        
        % ----------------------
        
        function outputResult = runner(obj)
            % This method contains code to get images in real time from
            % a webcam.
            % -------------------------------------
            fprintf('\n Enabling webcam-> \n');
            obj.camera = webcam(1);
            fprintf('\n -------------------------------- \n');
            fprintf('\n RUNNER Real time METHOD \n');
            fprintf('\n -------------------------------- \n');            
            % -------------------------------------
            % create a Window for real time video from camera
            window_results = figure('Name', 'Real Time Video');
            window_results.Position(3) = 2*window_results.Position(3);
            sub_window_image = subplot(1,2,1);
            sub_window_histogram = subplot(1,2,2);
            sub_window_histogram.PositionConstraint = 'innerposition';
                        
            while ishandle(window_results)
                % Display and classify the image
                fprintf('\n Working... put object to analyse-> \n');                
                % take an snapshop
                picture = obj.camera.snapshot;
                image(sub_window_image,picture);
                % image pre-processing for input in AlexNet
                %picture = imresize(picture,[227,227]);
                picture = imresize(picture, ConfigData.pixelSizeNeuralNetowrk);
                % -------------------------------------
                % classify images and put results on screen
                [label_predicted,score_predicted] = classify(obj.nnet, picture);
                title(sub_window_image,{char(label_predicted),num2str(max(score_predicted),2)});
                % --------------------------
                % select top five results from classification
                [~,idx] = sort(score_predicted,'descend');
                idx = idx(5:-1:1);
                classes = obj.nnet.Layers(end).Classes;
                classNamesTop = string(classes(idx));
                scoreTop = score_predicted(idx);
                % --------------------------
                % Draw histogram bar on windows
                barh(sub_window_histogram,scoreTop);
                title(sub_window_histogram,'Top 5');
                xlabel(sub_window_histogram,'Probability');
                xlim(sub_window_histogram,[0 1]);
                yticklabels(sub_window_histogram,classNamesTop);
                sub_window_histogram.YAxisLocation = 'right';
                % --------------------------
                drawnow
            end
            % -------------------------------------
            outputResult = 0;
        end        
        % -----------------------
    end
end


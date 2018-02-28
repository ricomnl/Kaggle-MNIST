%% MNIST Data Set

%% Initialization
clear ; close all; clc

%% Setup the parameters 
input_layer_size  = 784;  
hidden_layer_size = 25;   
num_labels = 10;  


% Load Training data
[X, y] = readData('train.csv');
y(y==0) = 10;

m = size(X, 1);

% randomly initialise weights
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

% Train NN
fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', 500);

lambda = 1;

costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                  hidden_layer_size, (input_layer_size + 1));
                  
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                  num_labels, (hidden_layer_size + 1));
                  
 
% Predict 

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

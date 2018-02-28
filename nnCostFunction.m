function [J grad] = nnCostFunction(nn_params, input_layer_size, ...
                                  hidden_layer_size, num_labels, X, y, lambda)
                                  
%% Cost function for a 2 layer network for classification
% returns cost and gradient

% reshape back into matrices
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
                  
% variables
m = size(X, 1);

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% recode Y (one hot encoding)
Y = zeros(m, num_labels);
for i = 1:m
  Y(i, y(i)) = 1;
end

% Feedforward
a1 = [ones(m, 1) X];
z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m, 1) a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);

% calculate penalty 
reg = (lambda/(2*m)) * (sum(sum(Theta1(:, 2:end) .^2)) + sum(sum(Theta2(:, 2:end) .^2)));

% calculate cost J
J = (sum(sum(-Y .* log(a3) - (1-Y) .* log(1-a3))))/m;
J += reg;

% calculate sigmas
sigma3 = (a3 .- Y);
sigma2 = (sigma3 * Theta2) .* sigmoidGradient([ones(size(z2,1),1) z2]);
sigma2 = sigma2(:, 2:end);

% accumulate gradients
delta1 = sigma2' * a1;
delta2 = sigma3' * a2;

% calculate regularized gradient
reg1 = (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
reg2 = (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];
Theta1_grad = delta1 ./ m + reg1;
Theta2_grad = delta2 ./ m + reg2;

% Unroll gradients 
grad = [Theta1_grad(:); Theta2_grad(:)];

end
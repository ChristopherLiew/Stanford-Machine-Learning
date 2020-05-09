function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% One Hot Encoding of y
y_encoded = zeros(m, num_labels); % 5000 by 10

for i = 1:m
    value = y(i);
    y_encoded(i, value) = 1;
end

% Add bias unit values to X
X = [ones(m, 1) X]; % 5000 by 401

% Forward Propagation for each obs and each class k
a_2 = sigmoid(Theta1 * X'); % 25 by 5000
a_2 = [ones(m,1) a_2']; % add in bias, 5000 by 26
a_3 = sigmoid(Theta2 * a_2'); % Output layer 10 by 5000

% Cost Function
J = (-1./m) .* sum(sum(y_encoded'.*log(a_3) + (1-y_encoded').*log(1-a_3)));

% Regularised Cost Function
% Drop Bias 
[~, t1_col] = size(Theta1);
[~, t2_col] = size(Theta2);
T1 = Theta1(:, (2:t1_col));
T2 = Theta2(:, (2:t2_col));
% Compute
J_reg = J + (lambda./(2.*m)).* (sum(sum(T1.^2)) + sum(sum(T2.^2)));
J = J_reg;

% -------------------------------------------------------------
% Backpropagation
for t = 1:m
    a_1 = X(t, :); % 1 by 401
    z_2 = Theta1 * a_1'; % 25 by 1
    a_2 = [ones(1) sigmoid(z_2)']; % 1 by 26 with bias 
    z_3 = Theta2 * a_2'; % 10 by 1
    a_3 = sigmoid(z_3); % 10 by 1
    
    z_2 = [1; z_2]; % add bias to z_2 in layer 2 (26 by 1)
    
    d_3 = a_3 - y_encoded(t, :)'; % 10 by 1
    d_2 = ((Theta2)'* d_3) .* sigmoidGradient(z_2); % 26 by 1 since l=2 has 26 neurons
    d_2 = d_2(2:end); % Drop bias 25 by 1
    
    % Updating Gradients
    Theta2_grad = Theta2_grad + d_3 * a_2; % 10 by 26
    Theta1_grad = Theta1_grad + d_2 * a_1; % 25 by 401
end

Theta2_grad = 1/m * Theta2_grad;
Theta1_grad = 1/m * Theta1_grad;
% =========================================================================

% With Regularisation
Theta2_grad(:, 2:end) = (Theta2_grad(:, 2:end)) + ((lambda/m)*Theta2(:, 2:end));
Theta1_grad(:, 2:end) = (Theta1_grad(:, 2:end)) + ((lambda/m)*Theta1(:, 2:end)); 

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end

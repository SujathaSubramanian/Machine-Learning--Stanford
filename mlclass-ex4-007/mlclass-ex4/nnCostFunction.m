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


y_matrix = eye(num_labels)(y,:);

X = [ones(m, 1) X];

a1 = X;
Z2 = a1*Theta1';
a2 = sigmoid(Z2);
a2 = [ones(m, 1) a2];
Z3 = a2* Theta2';
a3 = sigmoid(Z3);

a = -y_matrix.* log(a3);
b = (1-y_matrix).* log(1- a3);

J = (1/m) * sum(sum( a - b ));

N = hidden_layer_size;
K = input_layer_size;
O = num_labels;
theta1sqr = 0;
theta2sqr = 0;
newTheta1 = Theta1(:,[2:K+1]);
newTheta2 = Theta2(:,[2:N+1]);
for j = 1:N*K
	theta1sqr = theta1sqr + (newTheta1(j) * newTheta1(j));
end	

for j = 1:O*N
	theta2sqr = theta2sqr + (newTheta2(j) * newTheta2(j));
end

% -------------------------------------------------------------
reg = (lambda / (2*m)) * (theta1sqr + theta2sqr);
J = J + reg;
% =========================================================================

% Unroll gradients
%a2 = a2(:,[2:end]);
%a1 = a1(:,[2:end]);
d3 = a3 - (y_matrix);
d2 = (newTheta2' * d3') .* sigmoidGradient(Z2');
Delta2 = d3' * a2;
Delta1 = d2 * a1;
Theta1_grad = Delta1 * (1/m);
Theta2_grad = Delta2 * (1/m);
reg1 = ((lambda/m) * newTheta1);
reg2 = ((lambda/m)* newTheta2);
reg1 = [zeros(N, 1) reg1];
reg2 = [zeros(O, 1) reg2];
Theta1_grad = Theta1_grad + reg1;
Theta2_grad = Theta2_grad + reg2;

grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

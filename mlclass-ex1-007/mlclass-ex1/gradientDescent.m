function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values of the cost function (computeCost) and gradient here.
    %

h = X * theta;
R = h-y;

partialDerivative = X' * R;

theta = theta - (alpha * (1/m) * partialDerivative );


% ============================================================

% Save the cost J in every iteration    
J_history(iter) = computeCost(X, y, theta);
end





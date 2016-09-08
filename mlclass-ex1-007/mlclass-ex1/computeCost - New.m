function [J,grad] = computeCostNew(X, y, theta)

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

J = (1/m) * sum(-y'*log(sigmoid(X*theta)) - (1-y') * log(1-sigmoid(X*theta)));

grad = 0;
% =========================================================================

end



function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
R =0;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%



% You need to return the following variables correctly 
n= size(theta);
thetasqr = 0;
h = X * theta;
R = h-y;
R_sqr = R.^2;

Xj = X(:,[1])';
grad(1) = (1/m) * (sum(Xj * R));

for j = 2:n
	Xj= X(:,[j])';
	grad(j) = (1/m) * (sum(Xj * R)) + ((lambda/m) * theta(j)); 
	thetasqr = thetasqr + (theta(j) * theta(j));
end	
	

% =============================================================
regul = (lambda /(2 * m)) * thetasqr;
J = ((1/ (2 * m)) * sum(R_sqr)) + regul;

% =========================================================================

grad = grad(:);
end

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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h = X*theta;

error = h - y;

error_sqr = error.^2;

sum_sq_error = sum(error_sqr);

theta_sq = theta(2:end).^2;

sum_theta_sq = sum(theta_sq);


J =  ((1/(2*m)) * (sum_sq_error)) + ((lambda/(2*m))*sum_theta_sq);


sum_error = sum(error);

grad(1) = (1/m) * (sum(error)*X(1));



grad(2:end) = (1/m * (X(:,2:end)' * error))   + ((lambda/m)*theta(2:end));     %'

% n-1 x 1 				n-1 x m     m x 1

%%% 8/17: calculations seem to be working, and values match the 
%%% 	  solutions perfectly.  However, the submissions show errors...? revisit

% =========================================================================

grad = grad(:);

end

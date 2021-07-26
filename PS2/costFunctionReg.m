function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta




thtX = X*theta;

h_of_X = sigmoid(thtX);

theta_reg = theta(2:end);

J =  (1/m) * (-y' * log(h_of_X) - (1 - y)' * log(1 - h_of_X) ) + + ((lambda/(2*m)) * theta_reg' * theta_reg);


grad(1) = (1/m) * (( h_of_X - y)' * X(:,1));

grad(2:end) =  ( ((1/m) * (h_of_X - y)' * X(:,2:end)) + ((lambda/m) * theta_reg') )';

%	27 x 1			1 x 1		1 x 118		118 x 27		1 x 1		1 x 27



% =============================================================

end

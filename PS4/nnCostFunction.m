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

%%%Calculate Predictions%%%



%append a1 with a0^(1)
a1 = ones(m, 1);
a1 = [a1,X];

%calculate z2, a2
z2 = a1 * Theta1';      %'
a2 = sigmoid(z2);

%append a2 with a0^(2)
a02 = ones(rows(a2),1);
size(a02);
a2 = [a02 , a2];

%calculate z3, a3, h_theta_x
z3 = a2 * Theta2';      %' 
a3 = sigmoid(z3);

%a3 is a (m x num_labels) matrix.
%In the test case, 5000 x 10
h_of_X = a3;

%Now we have our predictions.

%_________

%%%Calculate Cost%%%

%Convert y into vectors of length num_labels
y_split = zeros(m,num_labels);

%Convert y values into binary vector notation e.g.[0,0,...1,0]
for i = 1:m
    y_split( i, y(i) ) = 1;
endfor

%Compute cost
for i = 1:m
    for k = 1:num_labels
        yki = y_split(i,k);
        h_of_Xik = h_of_X(i,k);
        costplus =  (-yki * log(h_of_Xik)) - ((1 - yki) * log(1 - h_of_Xik)) ; 
        J = J + (costplus);
    endfor
endfor

J = J/m;


%Compute regularization costplus

regsum1 = 0;
regsum2 = 0;

for j = 1:size(Theta1,1)
    for k = 2:size(Theta1,2)
        regsum1 = regsum1 + Theta1(j,k)^2;
    endfor
endfor

for j = 1:size(Theta2,1)
    for k = 2:size(Theta2,2)
        regsum2 = regsum2 + Theta2(j,k)^2;
    endfor
endfor

reg = (lambda/(2*m)) * (regsum1 + regsum2);

%Add regularization cost to the total cost function
J = J + reg;

%____________


%Backprop/Computing grad



for t = 1:m

    %1. fwd prop
    a1 = [ones(1,1), X(t,:)]';      %'
    %401 x 1

    z2 = Theta1 * a1;      %
    %25 x 1
    %   25 x 401,401 x 1

    a2 = [1; sigmoid(z2)];   
    %26 x 1

    z3 = Theta2 * a2;      %
    %10 x 1
    %    10 x 26,26 x 1

    a3 = sigmoid(z3);
    %10 x 1

    %2. calc. delta3 (output layer deltas)
    delta3 = a3 - y_split(t,:)'; %'
    %10 x 1

    %3. calc. delta2, remove first value
    delta2 = ( (Theta2'*delta3) .* sigmoidGradient([1;z2]) ); %'
    %26 x 1   26 x 10 , 10 x 1                 26 x 1

    %4. Calc cumulative gradient
    Theta2_grad = Theta2_grad + delta3 * a2';        %' 
    %10 x 26                   10 x 1  , 1 x 26    

    Theta1_grad = Theta1_grad + delta2(2:end) * a1';          %'
    %25 x 401       25 x 401     25 x 1       1 x 401

endfor

Theta2_grad = Theta2_grad/m;
Theta1_grad = Theta1_grad/m;


%Computing regularization
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m) * Theta2(:,2:end);
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m) * Theta1(:,2:end);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

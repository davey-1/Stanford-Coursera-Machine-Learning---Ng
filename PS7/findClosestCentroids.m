function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

%centroids is K x n matrix (# centroids x # features)
%X is a m x 1 matrix
%K is an int of # centroids
%idx is a m x 1 matrix logging the cluster assignment of each example


for i = 1:rows(X)

	%Xi is example # i in X
	Xi = X(i,:) ;

	%c_vec is K x 1 vector of costs associated with Xi and each centroid
	c_vec = zeros(K, 1);

	for k = 1:K

		len_err = norm( Xi - centroids(k,:));
		c_vec(k) = len_err.^2;

	endfor

	%ci is the index of the centroid that gives the lowest cost for Xi
	[c_val, ci] = min(c_vec);

	%set index of example i to ci
	idx(i) = ci;

endfor



% =============================================================

end


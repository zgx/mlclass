function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
%{
C_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';
sigma_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';

[T1,T2] = meshgrid(C_vec, sigma_vec);
cs = [T1(:) T2(:)];
l = length(cs);
rs = ones(l,1);

%plot3(C_vec, sigma_vec,rs);% C_vec .+ sigma_vec);
%plot3(cs(:,1), cs(:,2), rs);
%contour(cs(:,1), cs(:,2), cs(:,1) + cs(:,2));

for i = 1:l
	fprintf('train no. %d\n', i);
	model = svmTrain(X, y, cs(i,1), @(x1, x2) gaussianKernel(x1, x2, cs(i, 2)));
	preds = svmPredict(model, Xval);
	rs(i) = mean(double(preds ~= yval));
end;

[w,index] = min(rs);
C = cs(index,1);
sigma = cs(index, 2);

fprintf('the C:%f, sigma:%f\n', C, sigma);
%}
C = 0.3;
sigma = 0.1;
% =========================================================================

end

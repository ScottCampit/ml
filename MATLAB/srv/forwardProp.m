function yhat = forwardProp(X, beta, intercept)
%% FORWARDPROP Forward propagation for linear regression
% Detailed explanation of this function.
    yhat = sum(X .* beta' + intercept, 2);
end
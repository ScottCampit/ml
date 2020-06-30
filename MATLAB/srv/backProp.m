function [beta, intercept] = backProp(Xtrain, Ytrain, yhat, alpha, beta, intercept)
%% BACKPROP Backward propagation
% 
    % 1. Compute error for correction
    dB   = 2 / size(Xtrain, 1) .* -sum(Xtrain .* (Ytrain - yhat));
    dInt = 2 / size(Xtrain, 1) .* -sum(Ytrain - yhat);
    
    % 2. Use gradient descent for correction
    beta      = beta      - (alpha .* dB');
    intercept = intercept - (alpha .* dInt);
end
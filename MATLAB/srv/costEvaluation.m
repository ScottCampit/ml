function J = costEvaluation(Ytrain, yhat)
%% COSTEVALUATION Cost Evaluation
% To evaluate the cost for our linear regressor, we'll use the mean squared 
% error:
% 
% $$\mathrm{MSE}=\frac{1}{n}\sum_{i=1}^n {\left(y_i -\hat{y_i } \right)}^2$$
    J = 1 / size(Ytrain, 1) * sum((yhat-Ytrain) .^ 2);
end
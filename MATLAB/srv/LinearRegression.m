%% Linear Regression
% *Author*: Scott Campit
%% Summary
% This livescript is a tutorial to write up a linear regressor written from 
% scratch. We will use the |carbig| dataset built into MATLAB to predict MPG based 
% on all other numerical datasets that are not categoricals. First, we will build 
% a linear regressor using gradient descent. Then we will use the normal equation 
% to compute the regression coefficient. Finally, we'll use MATLAB's functions 
% to train a model.
%% Construct the regression dataset
% Let X be attributes that are useful to predict MPG. The continuous values 
% include acceleration, displacement, horsepower, and weight.

load carbig.mat
Y = MPG;
X = [Acceleration, Displacement, Horsepower, Weight];

% Normalize data using mean = 0 and standard deviation = 1
mu = nanmean(X);
sigma = nanstd(X);
X = (X - mu) ./ sigma

% Remove NaNs
Y(any(isnan(X), 2), :) = [];
X(any(isnan(X), 2), :) = [];
X(any(isnan(Y), 2), :) = [];
Y(any(isnan(Y), 2), :) = [];
% Split into training and test datasets
% Next, we have to split the data into a training and test set. Let's use this 
% |trainTestSplit| function:
%%
% 
%   function [Xtrain, Ytrain, Xtest, Ytest] = ...
%       trainTestSplit(X, Y, trainingSize, randomState)
%       
%       % OPTIONAL ARGUMENTS
%       if nargin < 3
%           trainingSize = 0.8;
%       end
%       
%       if nargin < 4
%           % For reproducibility - if you want randomly shuffled data, turn this
%           % off.
%           rng('default');
%       else
%           rng(randomState);
%       end
%       
%       if istable(X)
%           X = table2array(X); 
%       end
%       
%       if istable(Y)
%           Y = table2array(Y); 
%       end
%       
%       % Shuffle the dataset using the cvpartition function
%       cvObj = cvpartition(size(X, 1), 'HoldOut', trainingSize);
%       idx = cvObj.test;
%       
%       % Split into training and test data based on training size specified
%       Xtest  = X(~idx, :); Xtrain = X(idx,  :);
%       Ytest  = Y(~idx, :); Ytrain = Y(idx,  :);
%       
%   end
%

trainingSize = 0.8;
randomState = 'default';
[Xtrain, Ytrain, Xtest, Ytest] = trainTestSplit(X, Y, trainingSize, randomState);
%% Initialize the regression coefficients and intercept to be small and random numbers
% First, we need to initialize the regression coefficients to be some small, 
% arbitrary value with the size of the number of predictors. The intercept needs 
% to be the number of samples in the dataset by one column.

beta      = randn([size(Xtrain, 2), 1]) * 0.001;
intercept = randn([size(Xtrain, 1), 1]) * 0.001;
% Define necessary training functions
% Included in this repository are some additional functions we need to run the 
% linear regressor. Here are the necessary components: training phase (forward 
% propagation), cost evaluation, and learning phase (backward propagation + gradient 
% descent). 
%%
% 
%   % Function 1: Forward Propagation or Training Phase
%   function yhat = forwardProp(X, beta, intercept)
%       yhat = sum(X .* beta' + intercept, 2);
%   end
%   
%   % Function 2: Cost evaluation
%   function J = costEvaluation(Ytrain, yhat)
%       J = 1 / size(Ytrain, 1) * sum((yhat-Ytrain) .^ 2);
%   end
%   
%   % Function 3: Backward Propagation or Learning Phase
%   function [beta, intercept] = backProp(Xtrain, Ytrain, yhat, alpha, beta, intercept)
%       % 1. Compute error for correction -> derivative is different from lecture
%       dB   = 2 / size(Xtrain, 1) .* -sum(Xtrain .* (Ytrain - yhat));
%       dInt = 2 / size(Xtrain, 1) .* -sum(Ytrain - yhat);
%       
%       % 2. Use gradient descent for correction
%       beta      = beta      - (alpha .* dB');
%       intercept = intercept - (alpha .* dInt);
%   end
%
%% Train the linear regressor
% The next bit of code utilized all of the functions above to train a linear 
% regressor. This for loop can be packaged into another function, but it is explicitly 
% written below.

% Intialize some hyperparameters
epochs = 1000;
alpha = 0.001;

% Linear regression using Gradient Descent
for i = 1:epochs
    yhat = forwardProp(Xtrain, beta, intercept);
    J(i) = costEvaluation(Ytrain, yhat);
    [beta, intercept] = backProp(Xtrain, Ytrain, yhat, ...
                                 alpha, beta, intercept);
end
% The MSE curve
% Let's see how the error or cost reduces with each iteration:

% Cost curve
figure; plot(1:epochs, J, 'color', 'r');
title('Cost with learning rate = 0.001')
xlabel('Number of epochs'); ylabel('Cost (J)')
%% 
% After 400 epochs, the model MSE plateaus. So the idea number of epochs would 
% be around 400 to reduce the risk of overfitting.
%% Evaluating the model
% To evaluate the model, let's look at the hold out pearson correlation coefficient, 
% then perform cross validation.
% Hold out metrics

ypred_val = sum(Xtest .* beta', 2);
mse = 1 / size(Ytest, 1) * sum((ypred_val - Ytest) .^ 2);
[pears, pval] = corr(ypred_val, Ytest);
% Cross validation metrics
% We'll use the following cross validation function to evaluate the model
%%
% 
%   function summary = crossValidate_lm(X, Y, kfold)
%       
%       % Train test split
%       [Xtrain, Ytrain, Xtest, Ytest] = trainTestSplit(X, Y, 0.8, 'default');
%       
%       % Train 10 models using k-fold cross validation
%       for k = 1:kfold
%           
%           % Randomly generate cross validation indices from the Ytrain variable.
%           idx = crossvalind('Kfold', size(Ytrain, 1), kfold);
%           
%           % Get cross validation data
%           Xtrain2 = Xtrain(idx ~= k, :);
%           Ytrain2 = Ytrain(idx ~= k);
%           Xtest2  = Xtrain(idx == k, :);
%           Ytest2  = Ytrain(idx == k);
%           
%           % Initialize values
%           beta      = randn([size(Xtrain, 2), 1]) * 0.001;
%           intercept = randn([size(Xtrain, 1), 1]) * 0.001;
%           
%           % Intialize some hyperparameters
%           epochs = 1000;
%           alpha = 0.001;
%           
%           % Linear regression using Gradient Descent
%           for k = 1:epochs
%               yhat = forwardProp(Xtrain, beta, intercept);
%               J(k) = costEvaluation(Ytrain, yhat);
%               [beta, intercept] = backProp(Xtrain, Ytrain, yhat, ...
%                                            alpha, beta, intercept);
%           end
%           
%           ypred_val2     = sum(Xtest2 .* beta', 2);
%           mse           = 1 / size(Ytest2, 1) * sum((ypred_val2 - Ytest2) .^ 2);
%           [pears, pval] = corr(ypred_val2, Ytest2);
%           
%           model{k}.beta      = beta;
%           model{k}.mse       = mse;
%           model{k}.pears     = pears;
%           model{k}.pval      = pval;
%   
%       end
%       
%       % Obtain the best performing model from k-fold cross validation. In
%       % this case, we are choosing the model with the smallest MSE
%       [~, idx] = min([model.mse]);
%       finalModel = model{idx};
%       
%       % Compute final prediction using the best model
%       ypred_val     = sum(Xtest .* finalModel.beta', 2);
%       
%       % Get model metrics
%       mse           = 1 / size(Ytest2, 1) * sum((ypred_val2 - Ytest2) .^ 2);
%       [pears, pval] = corr(ypred_val, Ytest);
%       
%       % Save everything into a MATLAB structure
%       summary.name  = "Linear regressor summary";
%       summary.mse   = mse;
%       summary.beta  = finalModel.beta;
%       summary.pears = pears;
%       summary.pval  = pval;
%   end
%

kfold = 5;
summary = crossValidate_lm(X, Y, kfold)
%% Interpreting the linear regressor on the cars dataset

summary.beta
%% 
% To wrap things up, the pearson correlation coefficient is 0.82 and the p-value 
% of the pearson correlation coefficient w.r.t a normal distribution is 6.39E-21. 
% This means that the model captures the relationship betwen MPG and the acceleration, 
% displacement, horsepower, and weight for each car with a strong positive correlation. 
% 
% By examining the beta cofficients, acceleration is positively correlated with 
% MPG, while displacement, horsepower, and weight are negatively correlated with 
% MPG. Thus, if you want to maximize the MPG, you would want a car that can accelerate 
% quickly, but has low horsepower and weight. 
%% Using the normal equation
% Of course, there's another way to compute the regression coefficients by using 
% the *normal equation*:
% 
% $$\beta^ˆ ={\left(X^T X\right)}^{-1} X^T y$$

% Use normal equation
[Xtrain, Ytrain, Xtest, Ytest] = trainTestSplit(X, Y, trainingSize, randomState);
beta_normal = inv(Xtrain' * Xtrain) * (Xtrain' * Ytrain)
% Evaluate normal equation
ypred_val = sum(Xtest .* beta_normal', 2)
mse = 1 / size(Ytest, 1) * sum((ypred_val - Ytest) .^ 2)
[pears, pval] = corr(ypred_val, Ytest)
%% 
% As you can see, we get very similar performances using both models.
%% Using |fitlm| from MATLAB
% Now that we have ran through an algorithmic understanding of linear regression, 
% and the math required to solve the regression coefficients from a learning and 
% analytic perspective, let's see how we can fit a linear regressor using MATLAB's 
% functions, which will most likely be the way you use these tools. However, hopefully 
% from this exercise, you have a better understanding on how a regressor works 
% under the hood, as we will tweak these parameters in our next lecture. 

[Xtrain, Ytrain, Xtest, Ytest] = trainTestSplit(X, Y, trainingSize, randomState);
mdl = fitlm(Xtrain, Ytrain)
%% 
% The estimate is the regression coefficient, the standard error is the measure 
% of the standard deviation associated with each regression coefficient, the T 
% score is where the regression coefficient lies in a normal distribution and 
% how far it is from 0, and the two-tailed p-value tests the hypothesis that the 
% coefficient is equal to 0 or not.
% 
% This summary suggests that horsepower and weight are significant contributors 
% to predicting MPG. Further, all features from this algorithm show that there 
% is actually a negative relationship between each feature and MPG. This does 
% differ from our previous analysis, and does suggest that acceleration is not 
% a great indicator of MPG, which is reflected in the p-value
%% 
% Let's now use the model for making a prediction and evaluate the prediction.

ypred = predict(mdl, Xtest)
mse = 1 / size(Ytest, 1) * sum((ypred - Ytest) .^ 2)
[pears, pval] = corr(ypred, Ytest)
%% 
% As you can see, the built-in function does outperform our solutions, most 
% likely to some additional optimization steps MATLAB performs under the hood, 
% or due to random chance. However, both methods perform similarly.
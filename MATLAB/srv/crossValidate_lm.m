function summary = crossValidate_lm(X, Y, kfold)
%% CROSSVALIDATE_LM K-fold cross validation
    
    % Train test split
    [Xtrain, Ytrain, Xtest, Ytest] = trainTestSplit(X, Y, 0.8, 'default');
    
    % Train 10 models using k-fold cross validation
    for k = 1:kfold
        
        % Randomly generate cross validation indices from the Ytrain variable.
        idx = crossvalind('Kfold', size(Ytrain, 1), kfold);
        
        % Get cross validation data
        Xtrain2 = Xtrain(idx ~= k, :);
        Ytrain2 = Ytrain(idx ~= k);
        Xtest2  = Xtrain(idx == k, :);
        Ytest2  = Ytrain(idx == k);
        
        % Initialize values
        beta      = randn([size(Xtrain, 2), 1]) * 0.001;
        intercept = randn([size(Xtrain, 1), 1]) * 0.001;
        
        % Intialize some hyperparameters
        epochs = 1000;
        alpha = 0.001;
        
        % Linear regression using Gradient Descent
        for l = 1:epochs
            yhat = forwardProp(Xtrain, beta, intercept);
            J(l) = costEvaluation(Ytrain, yhat);
            [beta, intercept] = backProp(Xtrain, Ytrain, yhat, ...
                                         alpha, beta, intercept);
        end
        
        ypred_val2     = sum(Xtest2 .* beta', 2);
        mse           = 1 / size(Ytest2, 1) * sum((ypred_val2 - Ytest2) .^ 2);
        [pears, pval] = corr(ypred_val2, Ytest2);
        
        model{k}.beta      = beta;
        model{k}.mse       = mse;
        model{k}.pears     = pears;
        model{k}.pval      = pval;
    end
    
    % Obtain the best performing model from k-fold cross validation. In
    % this case, we are choosing the model with the smallest MSE
    for m = 1:length(model)
        mse(m) = model{m}.mse;
    end
    [~, idx] = min(mse);
    finalModel = model{idx};
    
    % Compute final prediction using the best model
    ypred_val     = sum(Xtest .* finalModel.beta', 2);
    
    % Get model metrics
    mse           = 1 / size(Ytest2, 1) * sum((ypred_val2 - Ytest2) .^ 2);
    [pears, pval] = corr(ypred_val, Ytest);
    
    % Save everything into a MATLAB structure
    summary.name  = "Linear regressor summary";
    summary.mse   = mse;
    summary.beta  = finalModel.beta;
    summary.pears = pears;
    summary.pval  = pval;
end
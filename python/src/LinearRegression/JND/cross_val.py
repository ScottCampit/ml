def cross_val(training_input, training_output, k_folds):
    from sklearn.utils import shuffle
    from linear_regressor import linear_regressor
    from numpy import zeros, argmax, size, concatenate

    X, y = shuffle(training_input, training_output)
    num_samples = size(training_input, 0)
    num_samples_bin = int(num_samples/k_folds)
    errors = zeros(k_folds)
    models = zeros([2, k_folds])
    R2s = zeros(k_folds)

    for i in range(k_folds):
        X_test = X[i * num_samples_bin:(i + 1) * num_samples_bin]
        X_train = concatenate((X[0:i * num_samples_bin], X[(i + 1) * num_samples_bin:]), axis=0)
        y_test = y[i * num_samples_bin:(i + 1) * num_samples_bin]
        y_train = concatenate((y[0:i * num_samples_bin], y[(i + 1) * num_samples_bin:]), axis=0)

        B, B0 = linear_regressor(X_train, y_train, epochs=10000)
        y_pred = B*X_test + B0

        MSE = sum(((y_pred-y_test)**2)/len(X_test))
        SSE = sum((y_pred-y_test)**2)
        y_bar = sum(y)/len(X_test)
        SST = sum((y-y_bar)**2)
        R2 = 1-SSE/SST

        errors[i] = MSE
        models[0, i] = B
        models[1, i] = B0
        R2s[i] = R2

    index = argmax(1-errors)
    return models[0, index], models[1, index], errors[index], R2s[index]


def linear_regressor(X_train, y_train, epochs):

    n = len(X_train)
    alpha = 0.02
    B = 0
    B0 = 0

    for i in range(epochs):
        y_pred = B*X_train + B0
        dB = sum(X_train*(y_pred - y_train))/n
        dB0 = sum((y_pred - y_train))/n
        B = B - alpha*dB
        B0 = B0 - alpha*dB0

    return B, B0

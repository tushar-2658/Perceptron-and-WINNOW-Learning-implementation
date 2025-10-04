
# modify alpha variable to change the promotion or demotion factor
# modify 'X' list to change the input features
# modfiy the 'y' list to change the target outputs
# modify theta variable to change the threshold
import numpy as np

alpha = 2
X = [
    [1, 0],
    [1, 1],
    [0, 0]
]
y = [0, 1, 0]
theta = 2
def winnow_activation(w_sum, theta): # returns 1 uf weighted sum > theta otherwise 0
    return 1 if w_sum > theta else 0
def train_winnow(X, y, alpha, theta, max_epochs = 1000):
    # X -> list of input vectors (no bias term)
    # y -> list of target outputs
    # alpha -> promotion or demotion factor
    # theta is the threshold value
    n_features = len(X[0])
    weights = np.ones(n_features) # init all weights to 1
    train_history = []
    # train loop start
    epoch = 0
    converge = False

    print(f"aplha : {alpha}\n Threshold : {theta}")
    while not converge and epoch < max_epochs:
        converge = True # assume covergence, set false when update occurs
        epoch += 1
        print(f"epoch : {epoch}")
        # iterating through each training example
        for i in range(len(X)):
            x_i = np.array(X[i])
            net = np.dot(weights, x_i) # calculate weighted sum
            prediction = winnow_activation(net, theta) # apply activation func
            error = y[i] - prediction
            if error != 0:
                if error > 0:
                    print(f"false negative detected")
                    for j in range(n_features):
                        if x_i[j] == 1:
                            weights[j] = weights[j] * alpha
                        elif error < 0:
                            for j in range(n_features):
                                if x_i[j] == 1:
                                    weights[j] = weights[j] / alpha
                print(f" new weights: {weights}")
            else:
                print(f"no update needed")
        predictions = [winnow_activation(np.dot(weights, np.array(x)), theta) for x in X]
        train_history.append((epoch, weights.copy(), predictions))
        print(f"\nEnd of epoch {epoch} - weights: {weights}")
    if not converge:
        print(f"did not converge after {max_epochs} epochs")
    return weights, train_history

def test_winnow(X, weights, theta):
    # x -> list of input vectors
    # weights -> trained weight vector
    # theta -> threshold value
    predictions = []
    for x in X:
        x_array = np.array(x)
        net = np.dot(weights, x_array)
        prediction = winnow_activation(net, theta)
        predictions.append(prediction)
    return predictions
if __name__ == "__main__":
    final_weights, history = train_winnow(X, y, alpha, theta)
    print("final weights:")
    for i in range(len(final_weights)):
        print(f"w{i + 1} (feature x{i + 1} weight): {final_weights[i]:.4f}")
    print(f"Threshold: {theta}")
    predictions = test_winnow(X, final_weights, theta)
    accuracy = sum([1 for i in range(len(y)) if predictions[i] == y[i]]) / len(y) * 100
    print(f"Training accuracy: {accuracy:.2f}%")
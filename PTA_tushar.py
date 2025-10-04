# modify the eta variable to change the learning rate
# modify the 'X' list to change input features
# modify the 'y' list to change target outputs

import numpy as np

eta = 0.4 # learning rate can be changed here
X = [ # training data
    [1,1,0],
    [1,1,1],
    [1,0,0]
]
y = [0,1,0] # target outputs
def perceptron_activation(w_sum): # activation func returns 1 when w_sum >= 0 else 0
    return 1 if w_sum >= 0 else 0

def train_perceptron(X, y, eta, max_epochs = 1000):
    # X -> list of input vectors
    # y -> list of target outputs
    # eta -> learning rate
    # max epochs -> maximum number of training epochs

    n_features = len(X[0])
    weights = np.zeros(n_features) # init weights to 0
    train_history = []
    # training loop start
    epoch = 0
    converge = False

    print(f"Learning rate (n): {eta}")
    print(f"number of training examples: {len(X)}")
    print(f"number of features : {n_features}")
    print(f"initial weights: {weights}")

    while not converge and epoch < max_epochs:
        converge = True # assume convergence, set false when any update comes
        epoch += 1
        print(f"\Epoch is {epoch}:")
        # iterate through training examples
        for i in range(len(X)):
            x_i = np.array(X[i])
            net = np.dot(weights, x_i) # calculate weighted sum
            prediction = perceptron_activation(net)
            error = y[i] - prediction

            print(f"example e{i + 1}: x={X[i]}, target={y[i]}")
            print(f"net input: {net:.4f}")
            print(f"Prediction: {prediction}")
            print(f"Error: {error}")

            if error != False: # update weights when there is an error
                converge = False
                # update weight rule : w = w + n * error * x
                w_update = eta * error * x_i
                weights = weights + w_update
                print(f"Weight update: {w_update}")
                print(f"New weights: {weights}")
            else:
                print(f"no update")
        #store current state
        predictions = [perceptron_activation(np.dot(weights, np.array(x))) for x in X]
        train_history.append((epoch, weights.copy(), predictions))
        if converge:
            print(f"convergent at {epoch}")
    if not converge:
        print(f"did not converge after {max_epochs} epochs")
    return weights, train_history


def test_perceptron(X, weights):
    # x -> list of input vectors
    # weights -> trained weight vector
    predictions = []
    for x in X:
        x_array = np.array(x)
        net = np.dot(weights, x_array)
        prediction = perceptron_activation(net)
        predictions.append(prediction)
    return predictions


if __name__ == "__main__":
    final_weights, history = train_perceptron(X, y, eta)
    print(f"Final weight values: ")
    print(f"w0 (bias weight): {final_weights[0]:.4f}")
    for i in range(1, len(final_weights)):
        print(f"w{i} (feature x{i} weight): {final_weights[i]:.4f}")
    print(f"full weight vector: {final_weights}")
    predictions = test_perceptron(X, final_weights)

    #print(f"predictions: {predictions}")

    accuracy = sum([1 for i in range(len(y)) if predictions[i] == y[i]]) / len(y) * 100
    print(f"training accuracy: {accuracy:.2f}%")
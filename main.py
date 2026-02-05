# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import time
from utils import load_mnist, training_curve_plot, calculate_performance_metrics, save_metrics_to_csv, visualize_model_weights, visualize_predictions

# ---------------------------------------------------------
# 1. INITIALIZATION & HELPER FUNCTIONS
# ---------------------------------------------------------

def initialize_network(K, D, D_i, D_o):
    """
    Initializes weights and biases for a network with K hidden layers and D neurons per layer.
    """
    all_weights = [None] * (K+1)
    all_biases = [None] * (K+1)

    if K == 0:
        all_weights[0] = np.random.randn(D_o, D_i) * np.sqrt(2/D_i)
        all_biases[0] = np.zeros((D_o, 1))
    else:
        all_weights[0] = np.random.randn(D, D_i) * np.sqrt(2/D_i)
        all_biases[0] = np.zeros((D, 1))
        
        for layer in range(1, K):
            all_weights[layer] = np.random.randn(D, D) * np.sqrt(2/D)
            all_biases[layer] = np.zeros((D, 1))
            
        all_weights[K] = np.random.randn(D_o, D) * np.sqrt(2/D)
        all_biases[K] = np.zeros((D_o, 1))
        
    return all_weights, all_biases

def initialize_momentum(all_weights, all_biases):
    """
    Initializes momentum velocities (v) to zeros with same shape as weights.
    """
    K = len(all_weights) - 1
    momentum_weights = [None] * (K + 1)
    momentum_biases = [None] * (K + 1)
    
    for layer in range(K + 1):
        momentum_weights[layer] = np.zeros_like(all_weights[layer])
        momentum_biases[layer] = np.zeros_like(all_biases[layer])
        
    return momentum_weights, momentum_biases

def random_mini_batches(X, Y, batch_size=64):
    """
    Creates a list of random mini-batches from (X, Y).
    """
    m = X.shape[1]
    mini_batches = []
    
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    for i in range(0, m, batch_size):
        mini_batch_X = shuffled_X[:, i : i + batch_size]
        mini_batch_Y = shuffled_Y[:, i : i + batch_size]
        mini_batches.append((mini_batch_X, mini_batch_Y))
    
    return mini_batches

# ---------------------------------------------------------
# 2. ACTIVATION FUNCTIONS
# ---------------------------------------------------------

def ReLU(preactivation):
    activation = preactivation.clip(0.0)
    return activation

def sigmoid(preactivation):
    activation = 1/(1+ np.exp(-preactivation))
    return activation

def indicator_function(x):
    x_in = np.array(x)
    x_in[x_in>0] = 1
    x_in[x_in<=0] = 0
    return x_in

def d_sigmoid(x):
    sig = sigmoid(x)
    x_dsig = sig * (1-sig)
    return x_dsig

def softmax(all_f):
    f_shifted = all_f - np.max(all_f, axis=0, keepdims=True)
    exp_f = np.exp(f_shifted)
    net_output = exp_f / np.sum(exp_f, axis=0, keepdims=True)
    return net_output

ACTIVATION_FUNCTIONS = {
    "ReLU": (ReLU, indicator_function),
    "Sigmoid": (sigmoid, d_sigmoid)
}

# ---------------------------------------------------------
# 3. CORE NETWORK FUNCTIONS
# ---------------------------------------------------------

def forward_pass(net_input, all_weights, all_biases, act_fn):
    K = len(all_weights) -1
    all_f = [None] * (K+1)
    all_h = [None] * (K+1)

    all_h[0] = net_input

    for layer in range(K):
        all_f[layer] = np.matmul(all_weights[layer], all_h[layer]) + all_biases[layer]
        all_h[layer+1] = act_fn(all_f[layer])

    all_f[K] = np.matmul(all_weights[K], all_h[K]) + all_biases[K]
    net_output = softmax(all_f[K])

    return net_output, all_f, all_h

def compute_cost(net_output, y):
    I = y.shape[1]
    epsilon = 1e-15
    return -np.sum(y * np.log(net_output + epsilon))/I

def d_cost_d_output(net_output, y):
    I = y.shape[1]
    return (net_output -y)/I

def backward_pass(all_weights, all_biases, all_f, all_h, y, net_output, d_act_fn):
    K = len(all_weights) - 1
    all_dl_dweights = [None] * (K+1)
    all_dl_dbiases = [None] * (K+1)
    all_dl_df = [None] * (K+1)
    all_dl_dh = [None] * (K+1)

    all_dl_df[K] = np.array(d_cost_d_output(net_output, y))

    layer_range = range(K, -1, -1)

    for layer in layer_range:
        all_dl_dbiases[layer] = np.sum(all_dl_df[layer], axis=1, keepdims=True)
        all_dl_dweights[layer] = np.matmul(all_dl_df[layer], all_h[layer].T)
        all_dl_dh[layer] = np.matmul(all_weights[layer].T, all_dl_df[layer])

        if layer > 0:
            all_dl_df[layer-1] = all_dl_dh[layer] * d_act_fn(all_f[layer-1])

    return all_dl_dweights, all_dl_dbiases

def update_parameters(all_weights, all_biases, all_dl_dweights, all_dl_dbiases, 
                                    momentum_weights, momentum_biases, learning_rate, beta):
    K = len(all_weights) - 1
    
    for layer in range(K + 1):
        # Weighted average of current gradient and past gradients
        momentum_weights[layer] = beta * momentum_weights[layer] + (1 - beta) * all_dl_dweights[layer]
        momentum_biases[layer] = beta * momentum_biases[layer] + (1 - beta) * all_dl_dbiases[layer]
        
        # Update parameters
        all_weights[layer] = all_weights[layer] - learning_rate * momentum_weights[layer]
        all_biases[layer] = all_biases[layer] - learning_rate * momentum_biases[layer]

def predict(X, Y, all_weights, all_biases, act_fn):
    net_output, _, _ = forward_pass(X, all_weights, all_biases, act_fn)
    predictions = np.argmax(net_output, axis=0)
    true_labels = np.argmax(Y, axis=0)
    accuracy = np.mean(predictions == true_labels)
    cost = compute_cost(net_output, Y)
    return predictions, accuracy, cost

def train_model(X_train, Y_train, X_test, Y_test, K, D, num_epochs, learning_rate, beta, batch_size, activation):
    start_time = time.time()
    
    act_fn, d_act_fn = ACTIVATION_FUNCTIONS[activation]
    
    D_i = X_train.shape[1] 
    D_o = Y_train.shape[1] 
    
    # Transpose to (Features, Samples)
    X_train_T = X_train.T
    Y_train_T = Y_train.T
    X_test_T = X_test.T
    Y_test_T = Y_test.T

    all_weights, all_biases = initialize_network(K, D, D_i, D_o)
    momentum_weights, momentum_biases = initialize_momentum(all_weights, all_biases)
    
    train_costs = []
    test_costs = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(num_epochs):
        batches = random_mini_batches(X_train_T, Y_train_T, batch_size)
        
        for mini_X, mini_Y in batches:
            net_output, all_f, all_h = forward_pass(mini_X, all_weights, all_biases, act_fn)
            all_dl_dweights, all_dl_dbiases = backward_pass(all_weights, all_biases, all_f, all_h, mini_Y, net_output, d_act_fn)
            update_parameters(all_weights, all_biases, all_dl_dweights, all_dl_dbiases, momentum_weights, momentum_biases, learning_rate, beta)
        
        # Monitoring
        _, train_acc, train_cost = predict(X_train_T, Y_train_T, all_weights, all_biases, act_fn)
        _, test_acc, test_cost = predict(X_test_T, Y_test_T, all_weights, all_biases, act_fn)
        
        train_costs.append(train_cost)
        test_costs.append(test_cost)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Cost {train_cost:.4f} | Test Acc {test_acc*100:.2f}%")
            
    elapsed_time = time.time() - start_time
    
    return all_weights, all_biases, train_costs, test_costs, train_accuracies, test_accuracies, elapsed_time

# ---------------------------------------------------------
# 4. MAIN EXECUTION
# ---------------------------------------------------------

if __name__ == "__main__":
    print("Loading Data...")
    X_train, Y_train, X_test, Y_test = load_mnist()

    # --- Hyperparameters ---
    K = 3
    D = 100
    num_epochs = 100
    learning_rate = 0.1
    beta = 0.9
    batch_size = 64
    activation_type = "ReLU"
    
    print(f"Training model with K={K} layers and D={D} units per layer...")
    
    # Train
    results = train_model(
        X_train, Y_train, X_test, Y_test, 
        K=K, 
        D=D, 
        num_epochs=num_epochs, 
        learning_rate=learning_rate, 
        beta=beta, 
        batch_size=batch_size, 
        activation=activation_type
    )

    weights_deep, biases_deep, train_c, test_c, train_a, test_a, elapsed_t = results
    
    print("Training complete....")

    # --- 1. Metrics Calculation ---
    
    # Run forward pass on Test Set to get probabilities
    act_fn, _ = ACTIVATION_FUNCTIONS[activation_type]
    test_net_output, _, _ = forward_pass(X_test.T, weights_deep, biases_deep, act_fn)
    
    # Calculate Precision, Recall, Accuracy
    final_acc, final_prec, final_rec = calculate_performance_metrics(test_net_output, Y_test.T)
    
    # Save to CSV
    csv_filename = "model_performance_metrics.csv"
    
    config_data = (
        "Linear" if K == 0 else "Deep_NN", 
        K, D, num_epochs, learning_rate, 
        batch_size, beta, activation_type
    )
    
    metrics_data = (final_acc, final_prec, final_rec)
    save_metrics_to_csv(csv_filename, config_data, metrics_data)
    
    print(f"Final Results -> Acc: {final_acc:.4f}, Prec: {final_prec:.4f}, Rec: {final_rec:.4f}")

    # --- 2. Plotting ---

    # Plot Training Curve
    training_curve_plot(
        "Model Performance", 
        train_c, test_c, train_a, test_a, 
        batch_size, learning_rate, beta, num_epochs, elapsed_t
    )

    # Visualize weights
    visualize_model_weights(weights_deep[0], title="Model Weights (Templates)")

    # Visualize predictions
    visualize_predictions(X_test, Y_test, test_net_output, title="Model Predictions")
    
    print("All plots saved.")
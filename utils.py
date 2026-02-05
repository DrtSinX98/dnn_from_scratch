import numpy as np
import imageio
import glob
import csv
import os
from matplotlib import pyplot as plt

def load_mnist():
    # Loads the MNIST dataset from png images
    #
    # Return
    # X_train - Training input 
    # Y_train - Training output (one-hot encoded)
    # X_test - Test input
    # Y_test - Test output (one-hot encoded)
    #
    # Each of them uses rows as data point dimension. Remember to transpose the output if you use columns for data point dimension
 
    NUM_LABELS = 10        
    # create list of image objects
    test_images = []
    test_labels = []    
    
    for label in range(NUM_LABELS):
        for image_path in glob.glob("MNIST/Test/" + str(label) + "/*.png"):
            image = imageio.imread(image_path)
            test_images.append(image)
            letter = [0 for _ in range(0,NUM_LABELS)]    
            letter[label] = 1
            test_labels.append(letter)  
            
    # create list of image objects
    train_images = []
    train_labels = []    
    
    for label in range(NUM_LABELS):
        for image_path in glob.glob("MNIST/Train/" + str(label) + "/*.png"):
            image = imageio.imread(image_path)
            train_images.append(image)
            letter = [0 for _ in range(0,NUM_LABELS)]    
            letter[label] = 1
            train_labels.append(letter)                  
            
    X_train= np.array(train_images).reshape(-1,784)/255.0
    Y_train= np.array(train_labels)
    X_test= np.array(test_images).reshape(-1,784)/255.0
    Y_test= np.array(test_labels)
    
    return X_train, Y_train, X_test, Y_test

def training_curve_plot(title, train_costs, test_costs, train_accuracy, test_accuracy, batch_size, learning_rate, beta, num_epochs, elapsed):
    lg=18
    md=13
    sm=9
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, y=1.15, fontsize=lg)    
    elapsed_min, elapsed_sec = divmod(elapsed, 60)
    sub = f'|  Batch size:{batch_size}  |  Learning rate:{learning_rate} | Momentum beta:{beta} | Number of Epochs:{num_epochs} | Training time: {elapsed_min:.0f} min {elapsed_sec:.1f} sec |'
    fig.text(0.5, 0.99, sub, ha='center', fontsize=md)
    x = np.array(range(1, len(train_costs)+1))*num_epochs/len(train_costs)
    axs[0].plot(x, train_costs, label=f'Final train cost: {train_costs[-1]:.4f}')
    axs[0].plot(x, test_costs, label=f'Final test cost: {test_costs[-1]:.4f}')
    axs[0].set_title('Costs', fontsize=md)
    axs[0].set_xlabel('Epochs', fontsize=md)
    axs[0].set_ylabel('Cost', fontsize=md)
    axs[0].legend(fontsize=sm)
    axs[0].tick_params(axis='both', labelsize=sm)
    
    axs[1].plot(x, train_accuracy, label=f'Final train accuracy: {100*train_accuracy[-1]:.2f}%')
    axs[1].plot(x, test_accuracy, label=f'Final test accuracy: {100*test_accuracy[-1]:.2f}%')
    axs[1].set_title('Accuracy', fontsize=md)
    axs[1].set_xlabel('Epochs', fontsize=md)
    axs[1].set_ylabel('Accuracy (%)', fontsize=sm)
    axs[1].legend(fontsize=sm)
    axs[1].tick_params(axis='both', labelsize=sm)
    
    # Save the plot
    filename = title.lower().replace(" ", "_") + ".png"
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved training curve: {filename}")

def calculate_performance_metrics(net_output, Y_one_hot):
    """
    Calculates Accuracy, Precision, and Recall given network output probabilities and ground truth.
    Uses Macro-Averaging for multi-class precision and recall.
    
    Arguments:
    net_output -- (D_o, N) array of output probabilities
    Y_one_hot -- (D_o, N) array of one-hot encoded ground truth
    """
    # 1. Convert Probabilities to Class Indices
    predictions = np.argmax(net_output, axis=0)
    true_labels = np.argmax(Y_one_hot, axis=0)
    
    # 2. Accuracy
    accuracy = np.mean(predictions == true_labels)
    
    # 3. Precision and Recall (Macro Average)
    num_classes = Y_one_hot.shape[0]
    precisions = []
    recalls = []
    
    for c in range(num_classes):
        # True Positives: Predicted c AND True c
        tp = np.sum((predictions == c) & (true_labels == c))
        
        # False Positives: Predicted c BUT True != c
        fp = np.sum((predictions == c) & (true_labels != c))
        
        # False Negatives: Predicted != c BUT True == c
        fn = np.sum((predictions != c) & (true_labels == c))
        
        # Precision c = TP / (TP + FP)
        # Handle division by zero if class never predicted
        p_c = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Recall c = TP / (TP + FN)
        # Handle division by zero if class doesn't exist in ground truth
        r_c = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        precisions.append(p_c)
        recalls.append(r_c)
        
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    
    return accuracy, macro_precision, macro_recall

def save_metrics_to_csv(filename, config, metrics):
    """
    Appends the model configuration and metrics to a CSV file.
    """
    file_exists = os.path.isfile(filename)
    
    # Define Column Headers
    headers = [
        "Model_Type", "K_Layers", "D_Neurons", "Epochs", "Learning_Rate", 
        "Batch_Size", "Momentum_Beta", "Activation", 
        "Test_Accuracy", "Test_Precision", "Test_Recall"
    ]
    
    # Combine config and metrics into one row
    row = list(config) + list(metrics)
    
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if file is created new
        if not file_exists:
            writer.writerow(headers)
            
        writer.writerow(row)
    
    print(f"Metrics saved to {filename}")

def visualize_model_weights(first_layer_weights, title="Visualized Weights"):
    """
    Visualizes the weights of the first layer and SAVES to PNG.
    """
    W = first_layer_weights

    plt.figure(figsize=(20, 4))
    for i in range(10):
        plt.subplot(1, 10, i + 1)
        img = W[i, :].reshape(28, 28)
        plt.imshow(img, cmap='viridis') 
        plt.title(f"Digit {i}")
        plt.axis('off')

    plt.suptitle(title, fontsize=16)
    filename = title.lower().replace(" ", "_") + ".png"
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {filename}")

def visualize_predictions(X_test, Y_test, test_net_output, title="Model Predictions"):
    """
    Visualizes 100 random predictions using pre-computed network output and SAVES to PNG.
    """
    num_samples = 100
    m_test = X_test.shape[0]
    indices = np.random.choice(m_test, num_samples, replace=False)

    # 1. Get Samples
    sample_X = X_test[indices]      # Shape (100, 784)
    sample_Y = Y_test[indices]      # Shape (100, 10)
    
    # 2. Get Predictions from the passed output
    # test_net_output is (10, N), so we slice columns
    sample_output = test_net_output[:, indices] # Shape (10, 100)

    predictions = np.argmax(sample_output, axis=0)
    true_labels = np.argmax(sample_Y, axis=1) # Axis 1 because sample_Y is (N, 10)
    
    # 3. Sort by Error
    matches = (predictions == true_labels)
    sorted_indices = np.argsort(matches)

    plt.figure(figsize=(10, 10)) 
    for i in range(num_samples):
        plt.subplot(10, 10, i + 1)
        plt.xticks([])
        plt.yticks([])

        # Get index relative to our batch of 100
        current_idx = sorted_indices[i]
        
        img = sample_X[current_idx].reshape(28, 28)
        plt.imshow(img, cmap=plt.cm.binary)

        is_correct = matches[current_idx]
        color = 'blue' if is_correct else 'red'
        predicted_label = predictions[current_idx]
        plt.text(0, 25, str(predicted_label), fontsize=18, color=color, fontweight='bold')

    plt.suptitle(title, fontsize=16)
    filename = title.lower().replace(" ", "_") + ".png"
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {filename}")
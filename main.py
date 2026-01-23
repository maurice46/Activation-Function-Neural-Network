import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random, math

random.seed(42) # reproducibility

# Activation Functions
def tanh(x): return math.tanh(x)
def d_tanh(x): return 1.0 - math.tanh(x) ** 2

def relu(x): return max(0.0, x)
def d_relu(x): return 1.0 if x > 0 else 0.0

def custom(x): return 0.1*x + math.tanh(x)
def d_custom(x): return 0.1 + (1.0 - math.tanh(x) ** 2)

# Helper Math Functions
def dot_product(inputs, weights): 
    return sum(inp * weight for inp, weight in zip(inputs, weights))

def vector_add(gradients, steps):
    return [gradient + step for gradient, step in zip(gradients, steps)]

def scalar_multiply(scalar, vector):
    return [scalar * v for v in vector]

def gradient_step(weights, gradient, step_size):
    step = scalar_multiply(step_size, gradient)
    return vector_add(weights, step)

# Activation Selector
def apply_activation(z, func):
    z = max(min(z, 30), -30) # clamp for numerical stability
    if func == 'tanh': return tanh(z)
    elif func == 'relu': return relu(z)
    elif func == 'custom': return custom(z)
    else: raise ValueError("Unknown activation")

def apply_activation_derivative(z, func):
    z = max(min(z, 30), -30)
    if func == "tanh": return d_tanh(z)
    elif func == "relu": return d_relu(z)
    elif func == "custom": return d_custom(z)
    else: raise ValueError("Unknown activation")

# Forward Pass
def feed_forward(network, input_vector, func):
    layer_activations, layer_zs = [], []
    current_input = input_vector
    
    for layer in network:
        input_with_bias = current_input + [1]
        linear_outputs = [dot_product(neuron, input_with_bias) 
                          for neuron in layer]
        activation_outputs = [apply_activation(linear_output, func) 
                              for linear_output in linear_outputs]
        layer_zs.append(linear_outputs)
        layer_activations.append(activation_outputs)
        current_input = activation_outputs

    return layer_zs, layer_activations

# Backpropagation (Squared Error Gradients)
def squared_error_gradients(network, input_vector, target_vector, func):
    layer_linear_outputs, layer_activations = feed_forward(network, input_vector, func)
    hidden_linear_outputs, output_linear_outputs = layer_linear_outputs
    hidden_activations, output_activations = layer_activations

    # output layer deltas
    output_deltas = []
    for activation, linear_output, target in zip(output_activations, output_linear_outputs, target_vector):
        delta = (activation - target) * apply_activation_derivative(linear_output, func)   # OUTPUT DELTA 
        output_deltas.append(delta)

    # hidden layer deltas 
    hidden_deltas = []
    for hidden_index, hidden_linear_output in enumerate(hidden_linear_outputs):
        downstream_error = 0.0
        for output_index, neuron in enumerate(network[1]):
            downstream_error += output_deltas[output_index] * neuron[hidden_index]
        delta = downstream_error * apply_activation_derivative(hidden_linear_output, func) # HIDDEN DELTA
        hidden_deltas.append(delta)

    # gradients 
    hidden_activation_with_bias = hidden_activations + [1]
    output_gradients = [[delta * activation for activation in hidden_activation_with_bias]
                        for delta in output_deltas]

    input_with_bias = input_vector + [1]
    hidden_gradients = [[delta * input_val for input_val in input_with_bias]
                        for delta in hidden_deltas]

    return [hidden_gradients, output_gradients]

# Data Cleaning / Test Splitting
df = pd.read_csv("winequality-red.csv", sep=";")
X_raw = df.drop('quality', axis=1).values
y_raw = (df['quality'] >= 6).astype(int).values 
X = StandardScaler().fit_transform(X_raw)
X_train, X_test, y_train_raw, y_test_raw = train_test_split(X, y_raw, test_size=0.2, random_state=42)

y_train = [[float(val)] for val in y_train_raw]
y_test  = [[float(val)] for val in y_test_raw]

# Network Architecture
NUM_INPUTS = X_train.shape[1]
NUM_HIDDEN = 10
NUM_OUTPUTS = 1
def init_network():
    hidden_layer = [[random.uniform(-1, 1) for _ in range(NUM_INPUTS + 1)] for _ in range(NUM_HIDDEN)]
    output_layer = [[random.uniform(-1, 1) for _ in range(NUM_HIDDEN + 1)] for _ in range(NUM_OUTPUTS)]
    return [hidden_layer, output_layer]

# Evaluate Training and Testing Accuracies 
def evaluate_on_data(network, X_data, y_data, func):
    correct = 0
    for x, y in zip(X_data, y_data):
        _, activations = feed_forward(network, x.tolist(), func)
        pred = 1 if activations[-1][0] >= 0.5 else 0
        if pred == int(y[0]):
            correct += 1
    return correct / len(X_data)

# Test Loss Function
def compute_test_loss(network, X_data, y_data, func):
    total = 0.0
    for x, y in zip(X_data, y_data):
        _, acts = feed_forward(network, x.tolist(), func)
        total += sum((a - t) ** 2 for a, t in zip(acts[-1], y))
    return total / len(X_data)

# Training
def train_network(network, func, lr, epochs, acc_threshold=0.75):
    losses, train_accs, test_accs = [], [], []
    threshold_epoch = None
    best_test_acc = 0.0
    
    for epoch in range(epochs):
        epoch_loss = 0.0

        # shuffling per epoch - improves generalization, stability, fair comparison across activations
        data = list(zip(X_train, y_train))
        random.shuffle(data)

        for x, y in data:
            gradients = squared_error_gradients(network, x.tolist(), y, func)

            network = [
                [gradient_step(neuron, grad, -lr)
                 for neuron, grad in zip(layer, layer_grads)]
                for layer, layer_grads in zip(network, gradients)
            ]

            _, activations = feed_forward(network, x.tolist(), func)
            predicted = activations[-1]
            epoch_loss += sum((a - t)**2 for a, t in zip(predicted, y))

        epoch_loss /= len(X_train)
        losses.append(epoch_loss)

        train_acc = evaluate_on_data(network, X_train, y_train, func)
        test_acc = evaluate_on_data(network, X_test, y_test, func)

        train_accs.append(train_acc)
        test_accs.append(test_acc)

        # track best test accuracy so far (current threshold)
        if test_acc > best_test_acc:
            best_test_acc = test_acc

        if test_acc >= acc_threshold and threshold_epoch is None:
            threshold_epoch = epoch + 1
    
    final_test_loss = compute_test_loss(network, X_test, y_test, func)

    return (
        network, 
        losses, 
        train_accs, 
        test_accs, 
        threshold_epoch,
        best_test_acc,
        final_test_loss
    )

# Experiments
activations = ["tanh", "relu", "custom"]
learning_rates = [0.001, 0.01, 0.1]
epochs = [100, 200, 500]
results = {}

for func in activations:
    print(f"\n================ {func.upper()} =================")
    for lr in learning_rates:
        for epoch in epochs:
            net = init_network()

            trained_net, losses, train_accs, test_accs, threshold_ep, best_test_acc, final_loss = \
                train_network(net, func, lr, epoch)
            
            print(
                f"lr={lr:<5} | Epochs={epoch:<4}| "
                f"Train Acc={train_accs[-1]:.4f} | "
                f"Test Acc={test_accs[-1]:.4f} | "
                f"Current Threshold:={best_test_acc:.4f} | "
                f"Threshold@0.75: {threshold_ep} | "
                f"Final Test Loss={final_loss:.4f}"
            )

            results[f"{func}_lr{lr}_ep{epoch}"] = {
                "losses": losses,
                "train_accs": train_accs,
                "test_accs": test_accs,
                "threshold_epoch": threshold_ep,
                "best_test_acc": best_test_acc,
                "final_test_loss": final_loss,
                "lr": lr,
                "epochs": epoch,
                "activation": func
            }

def facet(func, metric, ylabel, title):
    lrs = sorted(set(r["lr"] for r in results.values() if r["activation"] == func))
    epochs = sorted(set(r["epochs"] for r in results.values() if r["activation"] == func))
    fig, axes = plt.subplots(
        len(epochs),
        len(lrs),
        figsize=(4 * len(lrs), 3 * len(epochs)),
        sharex=True,
        sharey=True
    )

    for i, ep in enumerate(epochs):
        for j, lr in enumerate(lrs):
            r = results[f"{func}_lr{lr}_ep{ep}"]
            ax = axes[i, j]
            ax.plot(r[metric], linewidth=1.8)
            ax.grid(alpha=0.3)

            if i == 0: ax.set_title(f"lr={lr}")
            if j == 0: ax.set_ylabel(f"{ylabel}\n(ep={ep})")
    
    fig.suptitle(title, fontsize=14)
    fig.supxlabel("Epoch")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

for func in activations:
    facet(func, "losses", "Loss", f"{func.upper()} - Loss")
    facet(func, "train_accs", "Accuracy", f"{func.upper()} - Training Accuracy")
    facet(func, "test_accs", "Accuracy", f"{func.upper()} - Testing Accuracy")


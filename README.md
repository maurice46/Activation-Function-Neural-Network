# Exploring Activation Functions to Improve Gradient Flow and Stability in Neural Networks

A neural network built **entirely from scratch in Python** (no PyTorch, no TensorFlow) to test whether a custom activation function can outperform Tanh and ReLU on gradient flow, training stability, and generalization.

**Author:** Maurice Murillo · Computer Science Senior Seminar, University of Sioux Falls · November 2025
**Full paper:** [`Research_Paper.pdf`](./Research_Paper.pdf)

> **TL;DR:** The custom activation `0.1x + tanh(x)` achieved the highest testing accuracy (78.88%) and lowest final test loss of any activation tested, while remaining far more stable than ReLU — which failed to learn entirely at higher learning rates.

---

## Abstract

This project investigates whether a custom activation function can improve gradient flow and training stability in a neural network compared to common functions such as Tanh and ReLU. After implementing the core building blocks of a neural network — gradient descent, backpropagation, and loss functions — from scratch, the custom function is benchmarked against Tanh and ReLU on a supervised binary classification task using the Wine Quality dataset from the UCI Machine Learning Repository. Performance is compared on accuracy, convergence speed, and gradient behavior.

## The problem

Tanh and ReLU are the standard choices for activation functions, but both have well-documented failure modes:

- **Tanh** is smooth and zero-centered, which helps gradients flow evenly during backpropagation — but it saturates at ±1 for large inputs, causing the **vanishing gradient problem** and slowing or stalling learning.
- **ReLU** is simple and efficient, and avoids vanishing gradients in its active region — but it zeroes out all negative inputs, which can permanently kill neurons (**dying ReLU**), and is prone to **exploding gradients** under aggressive optimization.

This raises the question explored in this project: *can a custom activation function reduce the weaknesses of Tanh and ReLU while still maintaining stable gradient flow?*

## The custom activation function

```
custom(x) = 0.1x + tanh(x)
```

The idea is to keep what works about Tanh — a smooth, zero-centered nonlinearity that helps weights update evenly — while fixing its core weakness. Tanh's derivative approaches zero as `x` grows large in either direction, which is what causes vanishing gradients. Adding the small linear term `0.1x` means the function's derivative never fully flattens to zero, no matter how large or small the input gets. In principle, this:

- Avoids dying neurons by keeping negative values alive
- Avoids vanishing gradients by keeping the derivative away from zero
- Maintains stable gradient flow throughout training

## How it's built

Every component is implemented from first principles in pure Python — no autograd, no deep learning framework:

- **Forward pass** — manual weighted sums and layer-by-layer activation
- **Backpropagation** — gradients derived by hand via the chain rule on squared error loss
- **Gradient descent** — manual weight updates using the gradient step rule
- **Architecture** — 11 input features → 10-neuron hidden layer → 1 output neuron (binary classifier)

`pandas`, `scikit-learn`, and `matplotlib` are used only for data loading, splitting/scaling, and plotting — the network itself has no ML library dependencies.

## Dataset

[Wine Quality (red)](https://archive.ics.uci.edu/dataset/186/wine+quality) from the UCI Machine Learning Repository — 1,599 samples of Portuguese "Vinho Verde" red wine, with 11 physicochemical features (acidity, sugar, sulfates, alcohol, etc.).

The task is framed as **binary classification**: wines scoring ≥6 are labeled "good" (1), all others "bad" (0). Features are standardized before training, and the data is split 80/20 into training and test sets.

## Experiment design

Each activation function (Tanh, ReLU, Custom) was trained across a full grid of:
- **Learning rates:** 0.001, 0.01, 0.1
- **Epochs:** 100, 200, 500

with identical architecture, dataset split, and optimization procedure across all runs, so performance differences come from the activation function alone. Evaluation criteria included training accuracy, testing accuracy, convergence speed (epochs to reach an 80% accuracy threshold), and loss curve behavior.

## Results

### Best result per activation function

| Activation | Learning Rate | Epochs | Train Accuracy | Test Accuracy | Final Test Loss |
|---|---|---|---|---|---|
| Tanh | 0.001 | 100 | 74.75% | 76.88% | 16.81% |
| ReLU | 0.01 | 500 | 81.08% | 77.19% | 17.64% |
| **Custom** | 0.001 | 500 | 77.48% | **78.88%** | **16.74%** |

The custom activation function achieved the highest testing accuracy and lowest final test loss of any activation tested in this experiment.

### Stability across the full experiment grid

Looking beyond best-case results, aggregating across every run in [`results.txt`](./results.txt) tells a more complete story:

| Activation | Avg Test Acc | Avg Final Loss | Runs That Failed to Learn |
|---|---|---|---|
| **Custom** | 0.732 | 0.181 | **0 / 27** |
| Tanh | 0.742 | 0.182 | 0 / 27 |
| ReLU | 0.627 | 0.329 | **13 / 33 (39%)** |

*"Failed to learn" = the network collapsed to ~44% test accuracy, i.e. defaulted to predicting the majority class — observed almost exclusively at `lr=0.1` for ReLU.*

### Per-activation findings

**Tanh** trained smoothly and predictably, with loss dropping rapidly early on, but performance plateaued — additional training past ~100 epochs produced little to no improvement and occasionally slightly hurt test accuracy. This matches the expected vanishing-gradient behavior: the gradient becomes too small to keep improving the model.

**ReLU** was highly learning-rate dependent. At `lr=0.01` it trained efficiently with smooth, fast loss reduction. At `lr=0.1`, it failed to learn altogether — loss stayed flat and elevated, consistent with dying neurons and exploding gradients under aggressive optimization.

**Custom** showed the most consistent balance of stability and generalization. Loss continued to decline gradually even at high epoch counts rather than flattening early like Tanh, and testing accuracy kept improving with more training rather than overfitting — suggesting the linear term successfully prevented gradient saturation while preserving Tanh's stable, zero-centered behavior.

## Conclusion

For this architecture, dataset, and training setup, the custom activation function `0.1x + tanh(x)` achieved the best testing accuracy and lowest final loss, while training more reliably than ReLU and continuing to improve past the point where Tanh plateaued. These results support the idea that adding a small linear component to a smooth nonlinear activation can help maintain gradient flow without sacrificing stability — but they're specific to this controlled experiment, not a general claim that the custom function is universally better. Future work could test it across different datasets, deeper architectures, and other optimizers, and compare it against other modern activations like Leaky ReLU, GELU, or Swish.

## Project structure

```
.
├── main.py                  # Full implementation: network, training loop, experiments, plots
├── winequality-red.csv      # Dataset (UCI Wine Quality, red wine)
├── winequality.names        # Dataset documentation/attribute info
├── results.txt               # Logged output from experiment sweeps
├── Research_Paper.pdf       # Full written paper with background, methods, and analysis
└── README.md
```

## Running it

```bash
pip install pandas matplotlib scikit-learn
python main.py
```

This runs the full 3 activations × 3 learning rates × 3 epoch counts grid, prints a results table, and displays faceted plots of loss and accuracy for each activation function.

## References

1. Alzubaidi, L., Zhang, J., Humaidi, A.J. et al. Review of deep learning: concepts, CNN architectures, challenges, applications, future directions. *J Big Data* 8, 53 (2021). https://doi.org/10.1186/s40537-021-00444-8
2. Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Wine Quality [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C56S3T
3. Dubey, S. R., Singh, S. K., & Chaudhuri, B. B. (2022). Activation functions in deep learning: A comprehensive survey and benchmark. *Neurocomputing*, 503, 92–108. https://doi.org/10.1016/j.neucom.2022.06.111
4. Du, Larry. "All the Activation Functions." *DuBlog*, 8 July 2024, dublog.net/blog/all-the-activations/
5. Grus, Joel. *Data Science From Scratch*, 2nd Edition. O'Reilly Media, Inc, 2019.
6. Yang, L., Song, Q., Fan, Z., Liu, C., & Hu, M. (2023). Rethinking the activation function in a lightweight network. *Multimedia Tools and Applications*, 82(1), 1355–1371. https://doi.org/10.1007/s11042-022-13217-z
7. Z. Hu, J. Zhang, and Y. Ge, "Handling Vanishing Gradient Problem Using Artificial Derivative," in *IEEE Access*, vol. 9, pp. 22371-22377, 2021, doi: 10.1109/ACCESS.2021.3054915.

This project investigates whether a custom activation function can improve gradient flow and training 
stability in a neural network compared to common functions such as Tanh and ReLU. After reviewing some of the 
core concepts such as gradient descent, backpropagation, and loss functions, I compare these activation functions 
against my custom function on a supervised learning task, using a neural network implemented from scratch in 
Python using the Wine Quality dataset from the UC Irvine Machine Learning Repository. The goal is to create a 
binary classifier to predict good wine or bad wine. The performance of the activation functions is then compared on 
several criteria including accuracy, convergence speed, and gradient behavior.

A neural network is a computational model designed to learn from data and make accurate predictions. A neural network 
begins by computing the forward pass. This is the process of passing the input vector through the layers of the neural 
network to produce an output. Each neuron performs a mathematical transformation, and although it's simple for a single 
neuron, its complexity grows as layers are stacked sequentially. The forward pass is an essential part of the training 
process because it makes predictions of the model. This prediction is used to measure the error and update the parameters 
to make a better prediction in the next forward pass.  

The loss function determines how wrong the model prediction was compared to the true values. Without 
the loss function, the network would have no way of knowing if its predictions are improving or how its parameters 
should be adjusted.

After calculating the loss function, we know the errors in the network. Backpropagation is a gradient computation method 
used for training neural networks. It computes how much each weight contributed to the total error. The result is a set of
gradients (partial derivatives) of each weight. These gradients are used to update the weights so the network can make
accurate predictions.

An activation function is a mathematical function applied to each neuron’s weighted sum. It decides what 
information passes forward, how the gradients flow backwards during backpropagation, and how well the network 
learns. It decides whether the neurons activate and how strong they are. Activation functions make deep learning 
possible.

Although Tanh and ReLU are commonly used activation functions in neural networks, they introduce 
problems that can affect training performance. Tanh is smooth and produces zero-centered activations, but it causes 
vanishing gradients and can be expensive to compute. ReLU is simple, efficient, and fast, but it causes dying 
neurons and exploding gradients. Given these issues, we explore whether it is possible to design a custom activation 
function that reduces the weaknesses of Tanh and ReLU while still maintaining stability and gradient flow during 
training?

For the custom activation function, I want to bring in the best parts of Tanh. I think using Tanh is important 
because it gives a zero-centered output, which helps the network learn faster and keeps gradients flowing smoothly 
during backpropagation. The custom activation function is: 
                                          c𝑢𝑠𝑡𝑜𝑚(𝑥) = 0.1𝑥 + 𝑇𝑎𝑛ℎ(𝑥). 
This function keeps the output centered around zero, but it also fixes one of Tanh’s biggest weaknesses. 
When input becomes very large or very small, Tanh saturates at +1 or –1, which makes the derivatives extremely 
small. This slows down the learning process and makes training unstable. By adding the small linear term 0.1x, the 
function never fully flattens out, so the gradients never become zero. The function keeps the negative values (avoids 
dying neurons), keeps the output derivatives away from zero (avoids vanishing gradients), and keeps gradient flow 
stable during training (reduces changes of exploding gradients). Overall. I think this function will provide a good 
balance between non-linearity and strong gradients in the network. 

All activation functions were evaluated under identical conditions using multiple learning rates and epoch counts. 
For this experiment, the custom activation function achieved the highest testing accuracy and the lowest final loss, 
showing improved generalization performance relative to both Tanh and ReLU. While ReLU demonstrated strong 
training accuracy, it was more sensitive to the learning rate selection and demonstrated complete failure under 
aggressive optimization. Tanh provided stable learning behavior, but its performance was limited by gradients 
becoming too small and early plateaus during training.  

These results suggest that adding a small linear component to a smooth nonlinear activation function can 
help maintain gradient flow without sacrificing training stability. However, these findings are specific to this neural 
network architecture, dataset, and training setup. The results do not imply that the custom activation function is 
universally better, but instead show that it provides measurable benefits within this controlled experiment. Further 
work could extend this study by evaluating the custom activation function on different datasets, network 
architectures, and optimization methods beyond gradient descent, as well as comparing it to other modern activation 
functions. Nonetheless, this experiment demonstrates that the proposed custom activation function can improve 
learning stability and generalization compared to Tanh and ReLU within this specific network architecture and 
dataset. 


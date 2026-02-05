# Mathematical Formulation of Deep Neural Networks

**Author:** Pritish Joshi  
**Affiliation:** Department of Information Technology, Uppsala University, Uppsala, Sweden

---

## 1. General Neural Network Equation
A deep neural network is modeled as a recursive composition of affine transformations and non-linear activation functions. The network parameters are defined as $\phi = \{\beta_k, \Omega_k\}_{k=0}^K$, where $\beta_k$ denotes bias vectors and $\Omega_k$ denotes weight matrices connecting layer $k$ to layer $k+1$.

The general equation for the output $y$ is:
$$y = \beta_K + \Omega_K a\Big[\beta_{K-1} + \Omega_{K-1} a\big[\dots \beta_1 + \Omega_1 a[\beta_0 + \Omega_0 x] \dots \big]\Big]$$
where $a[\cdot]$ represents the activation function.

---

## 2. Forward Pass
The forward pass calculates the state of each layer sequentially. For a training example $x_i$, we define $f_k$ as pre-activation and $h_k$ as activation at layer $k$:

$$f_0 = \beta_0 + \Omega_0 x_i$$
$$h_1 = a[f_0]$$
$$f_k = \beta_k + \Omega_k h_k$$
$$h_{k+1} = a[f_k]$$

The final output is $y = f_K$.

---

## 3. Backward Pass (Backpropagation)
We compute gradients using the backpropagation algorithm, utilizing the chain rule recursively from the output layer backwards.

### Recursive Chain Rule
For any intermediate layer $k$, the gradient with respect to the pre-activation $f_{k-1}$ is:
$$\frac{\partial l_i}{\partial f_{k-1}} = \frac{\partial h_k}{\partial f_{k-1}} \cdot \Omega_k^T \cdot \frac{\partial l_i}{\partial f_k}$$

### ReLU Activation (Indicator Function)
If $h_k = \text{ReLU}[f_{k-1}]$, the derivative is the indicator function $\mathbb{I}[f_{k-1} > 0]$:
$$\frac{\partial l_i}{\partial f_{k-1}} = \mathbb{I}[f_{k-1} > 0] \odot \left( \Omega_k^T \frac{\partial l_i}{\partial f_k} \right)$$

### Sigmoid Activation
If $h_k = \sigma(f_{k-1})$, the derivative is $h_k \odot (1 - h_k)$:
$$\frac{\partial l_i}{\partial f_{k-1}} = \left( h_k \odot (1 - h_k) \right) \odot \left( \Omega_k^T \frac{\partial l_i}{\partial f_k} \right)$$

### Gradients w.r.t. Parameters
$$\frac{\partial l_i}{\partial \beta_k} = \frac{\partial l_i}{\partial f_k}$$
$$\frac{\partial l_i}{\partial \Omega_k} = \frac{\partial l_i}{\partial f_k} h_k^T$$

---

## 4. Optimization

### Stochastic Gradient Descent (SGD)
To minimize the cost $L[\phi]$, we update parameters using mini-batches $\mathcal{B}_t$. The update rule at iteration $t$ is:

$$\phi_{t+1} \leftarrow \phi_t - \alpha \cdot \sum_{i \in \mathcal{B}_t} \frac{\partial l_i[\phi_t]}{\partial \phi}$$

**Why SGD?**
* **Efficiency:** Faster than full batch gradient descent on large datasets.
* **Generalization:** Noise from mini-batches helps escape local minima.

### Momentum
Momentum smooths the optimization path by using a weighted average of gradients:

$$m_{t+1} \leftarrow \beta \cdot m_t + (1 - \beta) \sum_{i \in \mathcal{B}_t} \frac{\partial l_i[\phi_t]}{\partial \phi}$$
$$\phi_{t+1} \leftarrow \phi_t - \alpha \cdot m_{t+1}$$

**Why Momentum?**
* **Dampening:** Reduces oscillations in high-curvature directions.
* **Acceleration:** Speeds up convergence in consistent gradient directions.

---

## Usage

### 1. Installation

1. Clone the repository

```bash
git clone https://github.com/DrtSinX98/dnn_from_scratch.git && cd dnn_from_scratch
```

2. Ensure you have Python installed (version 3.8+ recommended). Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

### 2. Running the code

```bash
python main.py
```

### 3. Hyperparameter tuning

Edit the code and tune the hyperparameters in the 'Hyperparameters' section.


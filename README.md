PyTorch Math Fundamentals: XOR Neural Implementation
This repository contains a deep-dive implementation of a Multi-Layer Perceptron (MLP) designed to solve the XOR problem. The primary goal of this project is to deconstruct the "black box" of Deep Learning frameworks by manually implementing the core mathematical components of the training process.

Project Highlights
Unlike standard implementations, this project focuses on the underlying mathematics:

Manual Loss Function: Implementation of the Binary Cross-Entropy (BCE) formula using pure Tensor operations.

Manual Optimization: Weight updates performed via Stochastic Gradient Descent (SGD) by manipulating gradients directly, bypassing optimizer.step().

Geometric Insight: Visualization of the decision boundary to demonstrate how non-linear activation functions transform the input space.

Mathematical Foundations
1. The Non-Linearity (Sigmoid)

To solve non-linearly separable problems like XOR, we apply the Sigmoid activation function to map any real-valued number into the (0,1) interval:

σ(z)= 
1+e 
−z
 
1
​	
 
2. Binary Cross-Entropy (BCE)

The error measurement (Loss) is calculated manually following the Information Theory principle:

L=− 
N
1
​	
  
i=1
∑
N
​	
 [y 
i
​	
 ⋅log( 
y
^
​	
  
i
​	
 )+(1−y 
i
​	
 )⋅log(1− 
y
^
​	
  
i
​	
 )]
3. Weight Update (The Delta Rule)

Gradient descent is applied manually. Weights are updated by moving in the opposite direction of the gradient:

W 
new
​	
 =W 
old
​	
 −η⋅∇ 
W
​	
 L
Where η represents the Learning Rate.


Getting Started
Clone the repository:

Bash
git clone https://github.com/your-username/pytorch-math-fundamentals.git
Setup the environment:

Bash
python -m venv venv
source venv/bin/activate  # Linux/macOS

or

.\venv\Scripts\activate   # Windows
Install dependencies:

Bash
pip install torch matplotlib seaborn
Results & Visualization
The model achieves 100% accuracy. The plot below illustrates how the hidden layer creates a non-linear decision boundary to separate the XOR classes:

Tech Stack

- Python
- PyTorch (Tensors & Autograd)
- Matplotlib / Seaborn (Mathematical Visualization)

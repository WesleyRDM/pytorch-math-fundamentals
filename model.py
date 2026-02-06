import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

# Visualization of the decision boundary
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = torch.meshgrid(torch.arange(x_min, x_max, 0.01), 
                            torch.arange(y_min, y_max, 0.01), 
                            indexing='ij')    
    grid = torch.cat((xx.reshape(-1, 1), yy.reshape(-1, 1)), dim=1)
    
    model.eval()
    with torch.no_grad():
        Z = model(grid).reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx.numpy(), yy.numpy(), Z.numpy(), levels=50, cmap='RdBu', alpha=0.8)
    sns.scatterplot(x=X[:, 0].numpy(), y=X[:, 1].numpy(), hue=y.numpy().flatten(), 
                    palette='Set1', edgecolor='k', s=100)
    plt.title('Decision Boundary (XOR Problem)')
    plt.show()

# Generating synthetic data
def generate_data():
    X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
    y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)
    return X, y

def binary_cross_entropy_manual(y_pred, y_true):
    # Epsilon prevents log(0), avoiding NaN (mathematical indeterminacy)
    epsilon = 1e-15
    y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
    
    # Computing loss formula: -(y * log(p) + (1-y) * log(1-p))
    loss = -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
    return loss.mean()

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()        
        self.layer1 = nn.Linear(2, 4)
        self.layer2 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        return x

# Training using native PyTorch optimizer
def train_native(model, X, y, epochs=1000, lr=0.1):
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    print("--- Starting Native Training ---")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# Training using manual gradient updates (Mathematical deconstruction)
def train_manual(model, X, y, epochs=5000, lr=0.5):
    criterion = binary_cross_entropy_manual
    
    print("\n--- Starting Manual Training ---")
    for epoch in range(epochs):
        model.train()        
        output = model(X)
        loss = criterion(output, y)
        
        # Resetting gradients and performing Backward Pass
        model.zero_grad()
        loss.backward()                
        
        # Manually updating weights: w_new = w_old - lr * gradient
        with torch.no_grad():
            for param in model.parameters():
                param -= lr * param.grad
        
        if epoch % 1000 == 0:
            weight_sample = list(model.parameters())[0][0][0]
            print(f'Epoch {epoch}, Loss: {loss.item():.4f} | Weight Sample: {weight_sample.item():.6f}')

# Validation function
def validate(model, X, y):
    model.eval()
    with torch.no_grad():
        output = model(X)
        predicted = (output > 0.5).float()
        accuracy = (predicted == y).float().mean()
    print(f'\nFinal Accuracy: {accuracy.item()*100:.2f}%')

# --- Execution ---
X, y = generate_data()

# 1. Native Approach
model_native = NeuralNetwork()
train_native(model_native, X, y)
validate(model_native, X, y)

# 2. Manual Approach (Mathematical focus)
model_manual = NeuralNetwork()
train_manual(model_manual, X, y)
validate(model_manual, X, y)

# Visualization
plot_decision_boundary(model_manual, X, y)
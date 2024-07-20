import torch

# Set device to CPU
device = torch.device('cpu')

# Generate some sample data (replace with your actual data)
x = torch.arange(10, dtype=torch.float32, device=device).view(-1, 1)
y = 3 * x + 2 + torch.randn(10, dtype=torch.float32, device=device)

# Define model (linear regression)
class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegression().to(device)  # Move model to CPU device

# Define loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train the model (replace with actual training loop)
for epoch in range(100):
    # Forward pass, calculate loss
    y_pred = model(x)
    loss = criterion(y_pred, y)
    print(f"{epoch}")
    print(f"{loss}")

    # Backward pass, update weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Make predictions (replace with actual prediction logic)
predicted = model(torch.tensor([5.0], dtype=torch.float32, device=device).view(1, 1))
print(f"Predicted value for x=5: {predicted.item()}")

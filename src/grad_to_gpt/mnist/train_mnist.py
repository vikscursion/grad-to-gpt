import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader

# --- data: MNIST as tensors, served in shuffled mini-batches ---
tfm = torchvision.transforms.ToTensor()
train_ds = torchvision.datasets.MNIST("data", train=True, download=True, transform=tfm)
test_ds = torchvision.datasets.MNIST("data", train=False, download=True, transform=tfm)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=1000)
# each batch -> images (B, 1, 28, 28), labels (B,); flatten images to (B, 784)

# model parameters
torch.manual_seed(0)  # reproducible random init
W1 = (torch.randn(784, 128) * 0.01).requires_grad_(True)
b1 = torch.zeros(128, requires_grad=True)
W2 = (torch.randn(128, 10) * 0.01).requires_grad_(True)
b2 = torch.zeros(10, requires_grad=True)
params = [W1, b1, W2, b2]

# print([tuple(p.shape) for p in params])  # temporary shape check


# forward pass
def forward(images):
    x = images.view(images.shape[0], -1)
    h = torch.relu(x @ W1 + b1)
    logits = h @ W2 + b2
    return logits


# images, labels = next(iter(train_loader))
# print("logits shape:", forward(images).shape)

# loss = F.cross_entropy(forward(images), labels)
# print("loss:", loss.item())

lr = 0.1
for epoch in range(3):
    for images, labels in train_loader:
        logits = forward(images)  # forward
        loss = F.cross_entropy(logits, labels)  # loss

        for p in params:  # zero the grads
            p.grad = None
        loss.backward()  # backward -> fills p.grad

        with torch.no_grad():  # nudge each param downhill
            for p in params:
                p -= lr * p.grad
    print(f"epoch {epoch}  loss {loss.item():.3f}")

# --- test accuracy ---
correct = total = 0
with torch.no_grad():
    for images, labels in test_loader:
        preds = forward(images).argmax(dim=1)  # highest-scoring class per image
        correct += (preds == labels).sum().item()
        total += labels.shape[0]
print(f"test accuracy: {correct / total:.4f}")

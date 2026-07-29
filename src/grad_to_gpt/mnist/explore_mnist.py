import matplotlib.pyplot as plt
import torchvision

train = torchvision.datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)

print("training images:", len(train))
img, label = train[0]  # one example = (image tensor, integer label 0-9)
print("one image shape:", tuple(img.shape), " label:", label)

fig, axes = plt.subplots(1, 8, figsize=(10, 2))
for ax, (image, digit) in zip(axes, train, strict=False):
    ax.imshow(image.squeeze(), cmap="gray")  # (1,28,28) -> (28,28) for imshow
    ax.set_title(str(digit))
    ax.axis("off")
plt.tight_layout()
plt.savefig("data/mnist_sample.png", dpi=100)
print("saved data/mnist_sample.png")

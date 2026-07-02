"""Component 2: an MLP on MNIST in raw PyTorch.

YOU build this in phase 3. No nn.Sequential / high-level Trainer the first time —
write the loop by hand so every step stays visible:

    for epoch in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()      # predict what happens if you forget this
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()

Needs the deep-learning deps:  uv sync --extra dl
Done (see SPEC.md): >= ~97% test accuracy; loss curve matches a standard baseline.
"""

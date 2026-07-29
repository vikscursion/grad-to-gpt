from engine import Value
from nn import MLP

xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]

model = MLP(3, [4, 4, 1])

for step in range(100):
    # forward
    ypred = [model(x) for x in xs]

    # loss
    loss = Value(0.0)
    for yp, yt in zip(ypred, ys, strict=False):
        e = yp - yt
        loss = loss + e * e

    # zero grads
    for p in model.parameters():
        p.grad = 0.0

    # backward
    loss.backward()

    # update
    for p in model.parameters():
        p.data += -0.05 * p.grad

    if step % 10 == 0:
        print(step, round(loss.data, 4))

print("final preds:", [round(model(x).data, 2) for x in xs])
print("targets    :", ys)

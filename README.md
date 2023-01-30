# GateL0RD
This is a lightweight PyTorch implementation of GateL0RD, our RNN presented in ["Sparsely Changing Latent States for Prediction and Planning in Partially Observable Domains"](https://arxiv.org/abs/2110.15949).


We provide two variants of GateL0RD: `GateL0RD` can be used like a regular PyTorch `RNN`, whereas `GateL0RDCell` can be used like a PyTorch `RNNCell`. To install put ```gatel0rd.py``` into your working directory.

Generic example using `GateL0RD`:
```python
from gatel0rd import GateL0RD
#...
model = GateL0RD(input_size=input_dim, hidden_size=args.latent_dim, reg_lambda=args.lambda, output_size=output_dim)
optimizer = optim.Adam(model.parameters(), lr=args.lr) # optimizer of your choice
# ...
for X,Y in training_data:
    Y_hat, H, Theta = model.forward(X)
    optimizer.zero_grad()
    loss_task = F.mse_loss(Y_hat, Y) # loss of your choice
    loss = model.loss(loss_task, Theta)
    loss.backward()
    optimizer.step()
#...
```
A repository containing all experiments of the paper, including examples on how to use a `GateL0RDCell`, can be found [here](https://github.com/martius-lab/GateL0RD-paper).

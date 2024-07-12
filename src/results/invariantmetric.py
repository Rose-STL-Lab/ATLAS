import torch
L = torch.load('group_50.pt')
a = 0.05
lr = 1.0
J = torch.randn(4, 4, requires_grad=True)
optimizer = torch.optim.LBFGS([J], lr=lr)

def closure():
    optimizer.zero_grad()
    loss = 0.0
    for j in range(L.shape[0]):
        Lj = L[j]
        loss += torch.sum(torch.square(Lj.transpose(0, 1) @ J + J @ Lj))
    loss -= a * torch.linalg.norm(J)
    loss.backward()
    return loss

for i in range(1000):
    optimizer.step(closure)
#print(J)

torch.save(J, f"metric_local.pt")
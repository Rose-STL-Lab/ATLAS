import torch
import tqdm

a = torch.tensor([10., 1, 1,  0])
b = torch.tensor([1.,  1, 1, 10])

subspace =torch.tensor([
    [10, 1, 1, 10],
    [1, 1, 1, 10.]
]).float()

lie = torch.nn.Parameter(subspace)
torch.nn.init.normal_(lie)
optimizer = torch.optim.Adam([lie])

def p(s, debug=False):
    bc = s[3] / b[3]
    ac = s[2] - bc

    return torch.sum(torch.abs((ac * a + bc * b) - s))

for e in range(10000):
    average_loss = []
    # reg
    trace = torch.einsum('kd,kd->k', lie, lie)
    mag = torch.sqrt(trace / lie.shape[1])
    onorm = lie / mag.unsqueeze(-1)
    norm = torch.abs(onorm)

    ortho_factor = 1
    reg = ortho_factor * torch.sum(torch.abs(torch.triu(torch.einsum('bi,ci->bc', norm, norm), diagonal=1)))
    proj = 1 * (p(onorm[0]) + p(onorm[1]))
    loss = reg + proj

    average_loss.append(loss.detach().cpu())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(p(onorm[0], True), p(onorm[1], True), reg)
torch.set_printoptions(precision=9, sci_mode=False)
print((lie[0] / torch.max(torch.abs(lie[0]))).data)
print((lie[1] / torch.max(torch.abs(lie[1]))).data)

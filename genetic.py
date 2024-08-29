import torch
from config import Config
from utils import rmse, get_device

device = get_device()

class Genetic:
    def __init__(self, score_fn, num_pops, pop_size, man_dim, mutation_eps=0.05):
        self.score_fn = score_fn
        self.matrices = torch.zeros(num_pops, pop_size, man_dim, man_dim, device=device)
        self.mutation_eps = mutation_eps

        torch.nn.init.normal_(self.matrices, 0, 1)

    def print(self, gen_num, score):
        print("generation:", gen_num, "score:", score)
        print("representatives:", torch.mean(self.matrices, dim=1))

    def run(self, num_gens):
        for i in range(num_gens):
            scores = self.score_fn(self.matrices) 
            _, leader_inds = torch.min(scores, dim=1)
            parents = self.matrices[torch.arange(self.matrices.shape[0]), leader_inds]

            self.matrices[:] = parents.unsqueeze(1)
            rate = self.mutation_eps # / (i + 1)
            self.matrices += torch.randn(*self.matrices.shape, device=device) * rate

            # keep the parents
            self.matrices[:, 0] = parents

            self.print(i, torch.min(scores).detach())


if __name__ == '__main__':
    def norm_score(mat, vec):
        old = torch.linalg.vector_norm(vec.squeeze(-1), dim=-1)
        new = torch.linalg.vector_norm((mat @ vec).squeeze(-1), dim=-1)
        return torch.abs(new / old - 1)

    def score(mat):
        rand = torch.randn((2, 1), device=device)
        return norm_score(mat, rand)

    config = Config()
    genetic = Genetic(score, config.num_pops, config.pop_size, 2)

    genetic.run(100)


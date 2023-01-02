import torch
from math import log

points = torch.rand(150, 2, 
    pin_memory = True).cuda()

def reward_func(sol):
    rews = torch.zeros(len(sol)).cuda()
    pdist = torch.nn.PairwiseDistance(p=2)
    for i, row in enumerate(sol.argsort()):
        a = points[row]
        b = torch.cat((a[1:], a[0].unsqueeze(0)))
        rews[i] = pdist(a,b).sum()
    return rews

class FastCMA(object):
    def __init__(self, N, samples):
        self.samples = samples
        mu = samples // 2
        self.weights = torch.tensor([log(mu + 0.5)]).cuda()
        self.weights = self.weights - torch.linspace(
            start=1, end=mu, steps=mu).cuda().log()
        self.weights /= self.weights.sum()
        self.mueff = (self.weights.sum() ** 2 / (self.weights ** 2).sum()).item()
        # settings
        self.cc = (4 + self.mueff / N) / (N + 4 + 2 * self.mueff / N)
        self.c1 = 2 / ((N + 1.3) ** 2 + self.mueff)
        self.cmu = 2 * (self.mueff - 2 + 1 / self.mueff) 
        self.cmu /= ((N + 2) ** 2 + 2 * self.mueff / 2)
        # variables
        self.mean = torch.zeros(N).cuda()
        self.b = torch.eye(N).cuda()
        self.d = self.b.clone()
        bd = self.b * self.d
        self.c = bd * bd.T
        self.pc = self.mean.clone()

    def step(self, objective_f, step_size = 0.5):
        z = torch.randn(self.mean.size(0), self.samples).cuda()
        s = self.mean.view(-1, 1) + step_size * self.b.matmul(self.d.matmul(z))
        results = [{'parameters': s.T[i], 'z': z.T[i], 
        'fitness': f.item()} for i, f in enumerate(objective_f(s.T))]

        ranked_results = sorted(results, key=lambda x: x['fitness'])
        selected_results = ranked_results[0:self.samples//2]
        z = torch.stack([g['z'] for g in selected_results])
        g = torch.stack([g['parameters'] for g in selected_results])

        self.mean = (g * self.weights.unsqueeze(1)).sum(0)
        zmean = (z * self.weights.unsqueeze(1)).sum(0)
        self.pc *= (1 - self.cc)
        pc_cov = self.pc.unsqueeze(1) * self.pc.unsqueeze(1).T
        pc_cov = pc_cov + self.cc * (2 - self.cc) * self.c

        bdz = self.b.matmul(self.d).matmul(z.T)
        cmu_cov = bdz.matmul(self.weights.diag_embed())
        cmu_cov = cmu_cov.matmul(bdz.T)

        self.c *= (1 - self.c1 - self.cmu)
        self.c += (self.c1 * pc_cov) + (self.cmu * cmu_cov)
        self.d, self.b = torch.linalg.eigh(self.c, UPLO='U')
        self.d = self.d.sqrt().diag_embed()
        return ranked_results

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def plot_sol(rank):
    plot_data = points[rank.argsort()]
    plot_data = plot_data.detach().cpu().numpy()
    plt.figure(figsize=(12, 12))
    ax = pd.DataFrame(plot_data, columns=['x', 'y']).plot.line(x='x', y='y', legend=False)
    plt.savefig('sol', bbox_inches='tight', pad_inches=0)
    plt.close()      

with torch.no_grad():
    best_reward = None
    cma_es = FastCMA(N = len(points), samples=512)
    while True:
        try:
            res = cma_es.step(objective_f = reward_func)
        except Exception as e: 
            print(e)
            break
        if best_reward is None: best_reward = res[0]['fitness']
        if res[0]['fitness'] < best_reward:
            plot_sol(res[0]['parameters'])
            best_reward = res[0]['fitness']
            print("%i %f" % (epoch, best_reward))

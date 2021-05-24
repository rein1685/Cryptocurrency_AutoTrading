import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ACER():
    def __init__(self, model, memory, config):
        '''
        config parameters
        lr = learning rate
        K_epoch = K epoch
        gamma = gamma
        lmbda = lambda
        eps_clip = eps_clip
        '''
        self.model = model
        self.memory = memory
        self.config = config
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['lr'])

    def train(self, on_policy):
        s, a, r, prob, done_mask, is_first = self.memory.sample(on_policy)

        q = self.model.v(s)
        q_a = q.gather(1, a)
        pi = self.model.pi(s)
        pi_a = pi.gather(1, a)
        v = (q * pi).sum(1).unsqueeze(1).detach()

        rho = pi.detach() / prob
        rho_a = rho.gather(1, a)
        rho_bar = rho_a.clamp(max=self.config['c'])
        correction_coeff = (1 - self.config['c'] / rho).clamp(min=0)

        q_ret = v[-1] * done_mask[-1]
        q_ret_lst = []
        for i in reversed(range(len(r))):
            q_ret = r[i] + self.config['gamma'] * q_ret
            q_ret_lst.append(q_ret.item())
            q_ret = rho_bar[i] * (q_ret - q_a[i]) + v[i]

            if is_first[i] and i != 0:
                q_ret = v[i - 1] * done_mask[i - 1]  # When a new sequence begins, q_ret is initialized

        q_ret_lst.reverse()
        q_ret = torch.tensor(q_ret_lst, dtype=torch.float).unsqueeze(1)

        loss1 = -rho_bar * torch.log(pi_a) * (q_ret - v)
        loss2 = -correction_coeff * pi.detach() * torch.log(pi) * (q.detach() - v)  # bias correction term
        loss = loss1 + loss2.sum(1) + F.smooth_l1_loss(q_a, q_ret)

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
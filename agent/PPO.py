import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class PPO():
    def __init__(self, model, memory, config, device='cpu'):
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
        self.device = device

    def _make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []

        for transition in self.memory:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s, a, r, s_prime, done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float).to(self.device), torch.tensor(a_lst).to(self.device), \
                                              torch.tensor(r_lst).to(self.device), torch.tensor(s_prime_lst, dtype=torch.float).to(self.device), \
                                              torch.tensor(done_lst, dtype=torch.float).to(self.device), torch.tensor(prob_a_lst).to(self.device)

        self.memory.clean()

        return s, a, r, s_prime, done_mask, prob_a

    def train(self):
        s, a, r, s_prime, done_mask, prob_a = self._make_batch()

        for i in range(self.config['K_epoch']):
            td_target = r + self.config['gamma'] * self.model.v(s_prime) * done_mask
            td_target = td_target.float()

            delta = td_target - self.model.v(s)
            delta = delta.cpu().detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.config['gamma'] * self.config['lmbda'] * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.FloatTensor(advantage_lst).to(self.device)

            pi = self.model.pi(s)
            entorpy_bonus = Categorical(pi).entropy().mean()
            pi_a = pi.gather(1, a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.config['eps_clip'], 1 + self.config['eps_clip']) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.model.v(s), td_target.detach()) + entorpy_bonus*0.08

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
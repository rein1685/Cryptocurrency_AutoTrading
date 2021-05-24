import collections
import random
import torch

class ReplayBuffer():
    def __init__(self, buffer_limit, batch_size):
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.batch_size = 4

    def put(self, seq_data):
        self.buffer.append(seq_data)

    def sample(self, on_policy=False):
        if on_policy:
            mini_batch = [self.buffer[-1]]
        else:
            mini_batch = random.sample(self.buffer, self.batch_size)

        s_lst, a_lst, r_lst, prob_lst, done_lst, is_first_lst = [], [], [], [], [], []
        for seq in mini_batch:
            is_first = True  # Flag for indicating whether the transition is the first item from a sequence
            for transition in seq:
                s, a, r, prob, done = transition

                s_lst.append(s)
                a_lst.append([a])
                r_lst.append(r)
                prob_lst.append(prob)
                done_mask = 0.0 if done else 1.0
                done_lst.append(done_mask)
                is_first_lst.append(is_first)
                is_first = False

        s, a, r, prob, done_mask, is_first = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                             r_lst, torch.tensor(prob_lst, dtype=torch.float), done_lst, \
                                             is_first_lst
        return s, a, r, prob, done_mask, is_first

    def size(self):
        return len(self.buffer)
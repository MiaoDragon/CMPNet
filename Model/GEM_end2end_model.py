import torch.nn as nn
import torch
from Model.gem_utility import *
import numpy as np
import copy
# Auxiliary functions useful for GEM's inner optimization.
class End2EndMPNet(nn.Module):
    def __init__(self, total_input_size, AE_input_size, mlp_input_size, output_size, AEtype, \
                 n_tasks, n_memories, memory_strength, grad_step, CAE, MLP):
        super(End2EndMPNet, self).__init__()
        self.encoder = CAE.Encoder()
        self.mlp = MLP(mlp_input_size, output_size)
        self.mse = nn.MSELoss()
        self.opt = torch.optim.Adagrad(list(self.encoder.parameters())+list(self.mlp.parameters()))
        '''
        Below is the attributes defined in GEM implementation
        reference: https://arxiv.org/abs/1706.08840
        code from: https://github.com/facebookresearch/GradientEpisodicMemory
        '''
        self.margin = memory_strength
        self.n_memories = n_memories
        # allocate episodic memory
        #self.memory_data = torch.FloatTensor(
        #    n_tasks, self.n_memories, total_input_size)
        self.memory_input = torch.FloatTensor(
            n_tasks, self.n_memories, output_size*2)
        obs_size = [n_tasks, self.n_memories] + AE_input_size
        obs_size = tuple(obs_size)
        self.memory_obs = torch.zeros(obs_size).type(torch.FloatTensor)
        print('size of memory_obs:')
        print(self.memory_obs.size())
        self.memory_labs = torch.FloatTensor(n_tasks, self.n_memories, output_size)
        if torch.cuda.is_available():
            self.memory_input = self.memory_input.cuda()
            self.memory_obs = self.memory_obs.cuda()
            self.memory_labs = self.memory_labs.cuda()

        # allocate temporary synaptic memory
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        # edit: need one more dimension for newly observed data
        self.grads = torch.Tensor(sum(self.grad_dims), n_tasks+1)
        if torch.cuda.is_available():
            self.grads = self.grads.cuda()
        # allocate counters
        self.observed_tasks = []
        self.old_task = -1
        self.mem_cnt = np.zeros(n_tasks).astype(int)
        self.num_seen = np.zeros(n_tasks).astype(int)
        self.grad_step = grad_step
        self.total_input_size = total_input_size
        self.AE_input_size = AE_input_size
    def clear_memory(self):
        # set the counter to 0
        self.mem_cnt[:] = 0
        # set observed task to empty
        self.observed_tasks = []
        self.old_task = -1

    def set_opt(self, opt, lr=1e-2, momentum=None):
        # edit: can change optimizer type when setting
        if momentum is None:
            self.opt = opt(list(self.encoder.parameters())+list(self.mlp.parameters()), lr=lr)
        else:
            self.opt = opt(list(self.encoder.parameters())+list(self.mlp.parameters()), lr=lr, momentum=momentum)
    def forward(self, x, obs):
        # xobs is the input to encoder
        # x is the input to mlp
        #z = self.encoder(x[:,:self.AE_input_size])
        z = self.encoder(obs)
        mlp_in = torch.cat((z,x), 1)    # keep the first dim the same (# samples)
        return self.mlp(mlp_in)
    def loss(self, pred, truth):
        return self.mse(pred, truth)
    def pose_loss(self, pred, truth):
        # in 3d space
        pos_loss = self.mse(pred[:,:3], truth[:,:3])
        eps = 1e-4 # for numerical stability
        pred_ori = pred[:,3:]
        print('orientation:')
        print(pred_ori)
        pred_ori = pred_ori / pred_ori.norm(2, 2, True).clamp(min=eps).expand_as(pred_ori)
        print('normalized orientation:')
        print(pred_ori)
        ori_loss = self.mse(pred_ori, truth[:,3:])
        # weighted sum
        return pos_loss + ori_loss


    def load_memory(self, data):
        # data: (tasks, xs, ys)
        # continuously load memory based on previous memory loading
        tasks, xs, ys = data
        for i in range(len(tasks)):
            if tasks[i] != self.old_task:
                # new task, clear mem_cnt
                self.observed_tasks.append(tasks[i])
                self.old_task = tasks[i]
                self.mem_cnt[tasks[i]] = 0
            x = torch.tensor(xs[i])
            y = torch.tensor(ys[i])
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            self.remember(x, tasks[i], y)

    def remember(self, t, x, obs, y):
        # follow reservoir sampling
        # i-th item is remembered with probability min(B/i, 1)
        for i in range(len(x)):
            self.num_seen[t] += 1
            prob_thre = min(self.n_memories, self.num_seen[t])
            rand_num = np.random.choice(self.num_seen[t], 1) # 0---self.num_seen[t]-1
            if rand_num < prob_thre:
                # keep the new item
                if self.mem_cnt[t] < self.n_memories:
                    self.memory_input[t, self.mem_cnt[t]].copy_(x.data[i])
                    self.memory_obs[t, self.mem_cnt[t]].copy_(obs.data[i])
                    self.memory_labs[t, self.mem_cnt[t]].copy_(y.data[i])
                    self.mem_cnt[t] += 1
                else:
                    # randomly choose one to rewrite
                    idx = np.random.choice(self.n_memories, size=1)
                    idx = idx[0]
                    self.memory_input[t, idx].copy_(x.data[i])
                    self.memory_obs[t, idx].copy_(obs.data[i])
                    self.memory_labs[t, idx].copy_(y.data[i])

    '''
    Below is the added GEM feature
    reference: https://arxiv.org/abs/1706.08840
    code from: https://github.com/facebookresearch/GradientEpisodicMemory
    '''
    def observe(self, t, x, obs, y, remember=True, loss_f=self.loss):
        # remember: remember this data or not
        # update memory
        # everytime we treat the new data as a new task
        # compute gradient on all tasks
        # (prevent forgetting previous experience of same task, too)
        for _ in range(self.grad_step):

            if len(self.observed_tasks) >= 1:
                for tt in range(len(self.observed_tasks)):
                    if self.mem_cnt[tt] == 0 and tt == len(self.observed_tasks) - 1:
                        # nothing to train on
                        continue
                    self.zero_grad()
                    # fwd/bwd on the examples in the memory
                    past_task = self.observed_tasks[tt]
                    if tt == len(self.observed_tasks) - 1:
                        ptloss = loss_f(
                            self.forward(
                            self.memory_input[past_task][:self.mem_cnt[past_task]],
                            self.memory_obs[past_task][:self.mem_cnt[past_task]]),   # TODO
                            self.memory_labs[past_task][:self.mem_cnt[past_task]])   # up to current
                    else:
                        ptloss = loss_f(
                            self.forward(
                            self.memory_input[past_task],
                            self.memory_obs[past_task]),   # TODO
                            self.memory_labs[past_task])
                    ptloss.backward()
                    store_grad(self.parameters, self.grads, self.grad_dims,
                               past_task)

            # now compute the grad on the current minibatch
            self.zero_grad()
            loss = loss_f(self.forward(x, obs), y)
            loss.backward()

            # check if gradient violates constraints
            # treat gradient of current data as a new task (max of observed task + 1)
            # just to give it a new task label
            if len(self.observed_tasks) >= 1:
                # copy gradient
                new_t = max(self.observed_tasks)+1  # a new dimension
                store_grad(self.parameters, self.grads, self.grad_dims, new_t)
                indx = torch.cuda.LongTensor(self.observed_tasks) if torch.cuda.is_available() \
                    else torch.LongTensor(self.observed_tasks)   # here we need all observed tasks
                # here is different, we are using new_t instead of t to ditinguish between
                # newly observed data and previous data
                dotp = torch.mm(self.grads[:, new_t].unsqueeze(0),
                                self.grads.index_select(1, indx))
                if (dotp < 0).sum() != 0:
                    # remember norm
                    norm = torch.norm(self.grads[:, new_t], 2)
                    project2cone2(self.grads[:, new_t].unsqueeze(1),
                                  self.grads.index_select(1, indx), self.margin)
                    new_norm = torch.norm(self.grads[:, new_t], 2)
                    self.grads[:, new_t].copy_(self.grads[:, new_t] / new_norm * norm)
                    # before overwrite, to avoid gradient explosion, renormalize the gradient
                    # so that new gradient has the same l2 norm as previous
                    # it can be proved theoretically that this does not violate the non-negativity
                    # of inner product between memory and gradient
                    # copy gradients back
                    overwrite_grad(self.parameters, self.grads[:, new_t],
                                   self.grad_dims)
            self.opt.step()
        # after training, store into memory

        # when storing into memory, we use the correct task label
        # Update ring buffer storing examples from current task
        if remember:
            # only remember when the flag is TRUE
            if t != self.old_task:
                # new task, clear mem_cnt
                self.observed_tasks.append(t)
                self.old_task = t
                self.mem_cnt[t] = 0
            self.remember(t, x, obs, y)

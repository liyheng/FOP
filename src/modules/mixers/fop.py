import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FOPMixer(nn.Module):
    def __init__(self, args):
        super(FOPMixer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = int(np.prod(args.state_shape))
        self.action_dim = args.n_agents * self.n_actions
        self.state_action_dim = self.state_dim + self.action_dim
        self.n_head = args.n_head  
        self.embed_dim = args.mixing_embed_dim

        self.key_extractors = nn.ModuleList()
        self.agents_extractors = nn.ModuleList()
        self.action_extractors = nn.ModuleList()

        for i in range(self.n_head):  # multi-head attention
            self.key_extractors.append(nn.Linear(self.state_dim, 1)) 
            self.agents_extractors.append(nn.Linear(self.state_dim, self.n_agents))  
            self.action_extractors.append(nn.Linear(self.state_action_dim, self.n_agents)) 

        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states, actions=None, vs=None):
        bs = agent_qs.size(0)
        
        v = self.V(states).reshape(-1, 1).repeat(1, self.n_agents) / self.n_agents

        agent_qs = agent_qs.reshape(-1, self.n_agents)
        vs = vs.reshape(-1, self.n_agents)

        adv_q = (agent_qs - vs).detach()
        lambda_weight = self.lambda_weight(states, actions)-1

        adv_tot = th.sum(adv_q * lambda_weight, dim=1).reshape(bs, -1, 1)
        v_tot = th.sum(agent_qs + v, dim=-1).reshape(bs, -1, 1)

        return adv_tot + v_tot 

    def lambda_weight(self, states, actions): 
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.action_dim)
        state_actions = th.cat([states, actions], dim=1)

        head_keys = [k_ext(states) for k_ext in self.key_extractors]
        head_agents = [k_ext(states) for k_ext in self.agents_extractors]
        head_actions = [sel_ext(state_actions) for sel_ext in self.action_extractors]

        lambda_weights = []
        
        for head_key, head_agents, head_action in zip(head_keys, head_agents, head_actions):
            key = th.abs(head_key).repeat(1, self.n_agents) + 1e-10
            agents = F.sigmoid(head_agents)
            action = F.sigmoid(head_action)
            weights = key * agents * action
            lambda_weights.append(weights)
            
        lambdas = th.stack(lambda_weights, dim=1)
        lambdas = lambdas.reshape(-1, self.n_head, self.n_agents).sum(dim=1)

        return lambdas.reshape(-1, self.n_agents)
        

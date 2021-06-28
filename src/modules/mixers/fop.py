import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle as pkl


class FOPMixer(nn.Module):
    def __init__(self, args):
        super(FOPMixer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = int(np.prod(args.state_shape))
        self.action_dim = args.n_agents * self.n_actions
        self.state_action_dim = self.state_dim + self.action_dim
        self.unit_dim = 16
        self.n_head = args.n_head  
        self.embed_dim = args.mixing_embed_dim
        self.attend_reg_coef = args.attend_reg_coef

        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()
        self.keys = nn.ModuleList()
        self.agents_extractors = nn.ModuleList()
        self.action_extractors = nn.ModuleList()

        hypernet_embed = self.args.hypernet_embed
        for i in range(self.n_head):  # multi-head attention
            self.selector_extractors.append(nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                            nn.ReLU(),
                                            nn.Linear(hypernet_embed, self.embed_dim, bias=False)))  
            self.key_extractors.append(nn.Linear(self.unit_dim, self.embed_dim, bias=False))
            self.keys.append(nn.Linear(self.state_dim, 1)) 
            self.agents_extractors.append(nn.Linear(self.state_dim, self.n_agents))  
            self.action_extractors.append(nn.Linear(self.state_action_dim, self.n_agents)) 

        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states, actions=None, vs=None):
        bs = agent_qs.size(0)

        weight, attend_mag_regs = self.weight(agent_qs, states)
        weight = weight.reshape(-1, self.n_agents)  + 1e-10
        
        v = self.V(states).reshape(-1, 1).repeat(1, self.n_agents) / self.n_agents

        agent_qs = agent_qs.reshape(-1, self.n_agents)
        vs = vs.reshape(-1, self.n_agents)

        adv_q = (weight * (agent_qs - vs)).detach()
        lambda_weight = self.lambda_weight(states, actions)-1

        adv_tot = th.sum(adv_q * lambda_weight, dim=1).reshape(bs, -1, 1)
        v_tot = th.sum(weight * agent_qs + v, dim=-1).reshape(bs, -1, 1)

        return adv_tot + v_tot, attend_mag_regs

    def weight(self, agent_qs, states): # attention weights from Qatten
        states = states.reshape(-1, self.state_dim)
     
        unit_states = states[:, : self.unit_dim * self.n_agents]  
        unit_states = unit_states.reshape(-1, self.n_agents, self.unit_dim)
        unit_states = unit_states.permute(1, 0, 2)

        agent_qs = agent_qs.reshape(-1, 1, self.n_agents)  

        head_selectors = [sel_ext(states) for sel_ext in self.selector_extractors]
        head_keys = [[k_ext(enc) for enc in unit_states] for k_ext in self.key_extractors]

        # calculate attention per head
        head_attend_logits = []
        head_attend_weights = []
        for head_key, head_selector in zip(head_keys, head_selectors):
            attend_logits = th.matmul(head_selector.reshape(-1, 1, self.embed_dim),
                                      th.stack(head_key).permute(1, 2, 0))
            scaled_attend_logits = attend_logits / np.sqrt(self.embed_dim)

            attend_weights = F.softmax(scaled_attend_logits, dim=2)  
            head_attend_logits.append(attend_logits)
            head_attend_weights.append(attend_weights)

        head_attend = th.stack(head_attend_weights, dim=1)  
        head_attend = head_attend.reshape(-1, self.n_head, self.n_agents)

        head_attend = th.sum(head_attend, dim=1)

        # regularize magnitude of attention logits
        attend_mag_regs = self.attend_reg_coef * sum((logit ** 2).mean() for logit in head_attend_logits)

        return head_attend, attend_mag_regs

    def lambda_weight(self, states, actions):   # attention weights from Qplex
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.action_dim)
        state_actions = th.cat([states, actions], dim=1)

        head_keys = [k_ext(states) for k_ext in self.keys]
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
        

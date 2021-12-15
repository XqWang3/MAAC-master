import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain
from utils.misc import onehot_from_logits, categorical_sample

class AttentionCritic(nn.Module):
    """
    Attention network, used as critic for all agents. Each agent gets its own
    observation and action, and can also attend over the other agents' encoded
    observations and actions.
    """
    def __init__(self, sa_sizes, hidden_dim=32, norm_in=True, attend_heads=1):
        """
        Inputs:
            sa_sizes (list of (int, int)): Size of state and action spaces per
                                          agent
            hidden_dim (int): Number of hidden dimensions
            norm_in (bool): Whether to apply BatchNorm to input
            attend_heads (int): Number of attention heads to use (use a number
                                that hidden_dim is divisible by)
        """
        super(AttentionCritic, self).__init__()
        assert (hidden_dim % attend_heads) == 0
        self.sa_sizes = sa_sizes
        self.nagents = len(sa_sizes)
        self.attend_heads = attend_heads

        self.critic_encoders = nn.ModuleList()
        self.critics = nn.ModuleList()

        self.state_encoders = nn.ModuleList()
        # iterate over agents
        for sdim, adim in sa_sizes:
            idim = sdim + adim
            odim = adim
            encoder = nn.Sequential()
            if norm_in:
                encoder.add_module('enc_bn', nn.BatchNorm1d(idim,
                                                            affine=False))
            encoder.add_module('enc_fc1', nn.Linear(idim, hidden_dim))
            encoder.add_module('enc_nl', nn.LeakyReLU())
            self.critic_encoders.append(encoder)
            critic = nn.Sequential()
            critic.add_module('critic_fc1', nn.Linear(2 * hidden_dim,
                                                      hidden_dim))
            critic.add_module('critic_nl', nn.LeakyReLU())
            critic.add_module('critic_fc2', nn.Linear(hidden_dim, odim))
            self.critics.append(critic)

            state_encoder = nn.Sequential()
            if norm_in:
                state_encoder.add_module('s_enc_bn', nn.BatchNorm1d(
                                            sdim, affine=False))
            state_encoder.add_module('s_enc_fc1', nn.Linear(sdim,
                                                            hidden_dim))
            state_encoder.add_module('s_enc_nl', nn.LeakyReLU())
            self.state_encoders.append(state_encoder)

        attend_dim = hidden_dim // attend_heads
        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()
        self.value_extractors = nn.ModuleList()
        for i in range(attend_heads):
            self.key_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.selector_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.value_extractors.append(nn.Sequential(nn.Linear(hidden_dim,
                                                                attend_dim),
                                                       nn.LeakyReLU()))

        self.shared_modules = [self.key_extractors, self.selector_extractors,
                               self.value_extractors, self.critic_encoders]

    def shared_parameters(self):
        """
        Parameters shared across agents and reward heads
        """
        return chain(*[m.parameters() for m in self.shared_modules])

    def scale_shared_grads(self):
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        """
        for p in self.shared_parameters():
            p.grad.data.mul_(1. / self.nagents)

    def forward(self, inps, act=None, agents=None, return_softmax_act=False, return_q=False, return_trgt_q=False,
                return_max_q=False, regularize_pol=False,
                explore=False, return_all_q=False, regularize=False, return_log_pi=False, return_entropy=False,
                return_all_probs=False, return_attend=False, logger=None, niter=0):
        """
        Inputs:
            inps (list of PyTorch Matrices): Inputs to each agents' encoder
                                             (batch of obs + ac)
            agents (int): indices of agents to return Q for
            return_q (bool): return Q-value
            return_all_q (bool): return Q-value for all actions
            regularize (bool): returns values to add to loss function for
                               regularization
            return_attend (bool): return attention weights per agent
            logger (TensorboardX SummaryWriter): If passed in, important values
                                                 are logged
        """
        if agents is None:
            agents = range(len(self.critic_encoders))
        states = [s for s, a in inps]    # Xq: obs.shape: nagents * bs * s_dim
        actions = [a for s, a in inps]   # Xq:    .shape: nagents * bs * a_dim
        if act is not None:
            actions = act
        inps = [torch.cat((s, a), dim=1) for s, a in inps]
        # extract state-action encoding for each agent
        sa_encodings = [encoder(inp) for encoder, inp in zip(self.critic_encoders, inps)] # Xq: sa_enc.shape: nagents * bs * h_dim
        # extract state encoding for each agent that we're returning Q for
        s_encodings = [self.state_encoders[a_i](states[a_i]) for a_i in agents]  # Xq: sa_enc.shape: nagents * bs * h_dim
        # extract keys for each head for each agent
        all_head_keys = [[k_ext(enc) for enc in sa_encodings] for k_ext in self.key_extractors]
        # Xq: all_head_keys.shape: heads * nagents * bs * attend_dim
        # extract sa values for each head for each agent
        all_head_values = [[v_ext(enc) for enc in sa_encodings] for v_ext in self.value_extractors]
        # extract selectors for each head for each agent that we're returning Q for
        all_head_selectors = [[sel_ext(enc) for i, enc in enumerate(s_encodings) if i in agents]
                              for sel_ext in self.selector_extractors]
        # Xq: all_head_selectors.shape: heads * nagents * bs * attend_dim

        other_all_values = [[] for _ in range(len(agents))]
        all_attend_logits = [[] for _ in range(len(agents))]
        all_attend_probs = [[] for _ in range(len(agents))]
        # calculate attention per head
        for curr_head_keys, curr_head_values, curr_head_selectors in zip(
                all_head_keys, all_head_values, all_head_selectors):
            # iterate over agents
            for i, a_i, selector in zip(range(len(agents)), agents, curr_head_selectors):
                keys = [k for j, k in enumerate(curr_head_keys) if j != a_i]  # keys.shape: (nagents-1) * bs * attend_dim
                values = [v for j, v in enumerate(curr_head_values) if j != a_i]  # keys.shape: (nagents-1) * bs * attend_dim
                # calculate attention across agents
                # selector.shape: bs * attend_dim
                attend_logits = torch.matmul(selector.view(selector.shape[0], 1, -1),
                                             torch.stack(keys).permute(1, 2, 0))
                # attend_logits.shape: bs * 1 * (nagents-1) = torch.matmul( bs*1*attend_dim, bs*attend_dim*(nagents-1) )
                # scale dot-products by size of key (from Attention is All You Need)
                scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])
                attend_weights = F.softmax(scaled_attend_logits, dim=2)  # attend_weights.shape: bs * 1 * (nagents-1)
                other_values = (torch.stack(values).permute(1, 2, 0) *
                                attend_weights).sum(dim=2)   # other_values.shape: bs * attend_dim
                other_all_values[i].append(other_values)   # other_all_values.shape: nagents * bs * attend_dim
                all_attend_logits[i].append(attend_logits)
                all_attend_probs[i].append(attend_weights)
        # calculate Q per agent
        all_rets = []
        for i, a_i in enumerate(agents):
            head_entropies = [(-((probs + 1e-8).log() * probs).squeeze().sum(1)
                               .mean()) for probs in all_attend_probs[i]]
            agent_rets = []
            critic_in = torch.cat((s_encodings[i], *other_all_values[i]), dim=1)
            all_q = self.critics[a_i](critic_in)
            max_q = all_q.max(dim=1, keepdim=True)[0]
            int_acs = actions[a_i].max(dim=1, keepdim=True)[1]
            q = all_q.gather(1, int_acs)
            if return_softmax_act:
                probs = F.softmax(all_q, dim=1)
                on_gpu = next(self.parameters()).is_cuda
                if explore:
                    int_act, act = categorical_sample(probs, use_cuda=on_gpu)
                else:
                    act = onehot_from_logits(probs)
                agent_rets = [act]
            if return_log_pi or return_entropy:
                log_probs = F.log_softmax(all_q, dim=1)
            if return_log_pi:
                # return log probability of selected action
                agent_rets.append(log_probs.gather(1, int_act))
            if return_entropy:
                agent_rets.append(-(log_probs * probs).sum(1).mean())
            if return_all_probs:
                agent_rets.append(probs)
            if return_q:
                agent_rets.append(q)
            if return_trgt_q:
                agent_rets.append(all_q.gather(1, int_acs))
            if return_max_q:
                agent_rets.append(max_q)
            if return_all_q:
                agent_rets.append(all_q)
            if regularize:
                # regularize magnitude of attention logits
                attend_mag_reg = 1e-3 * sum((logit**2).mean() for logit in
                                            all_attend_logits[i])
                regs = (attend_mag_reg,)
                agent_rets.append(regs)
            if regularize_pol:
                agent_rets.append(1e-3 * (all_q ** 2).mean())
            if return_attend:
                agent_rets.append(np.array(all_attend_probs[i]))
            if logger is not None:
                # all_attend_probs[a_i].shape  head_num * bs * 1 * nagents-1
                logger.add_scalars('agent%i/attention' % a_i,
                                   dict(('head0_to_agent%i_weight' % h_i, weg) if h_i < a_i else
                                        ('head0_to_agent%i_weight' % (h_i+1), weg) for h_i, weg
                                        in enumerate(all_attend_probs[a_i][0][0].squeeze())),
                                   niter)
                logger.add_scalars('agent%i/attention' % a_i,
                                   dict(('head2_to_agent%i_weight' % h_i, weg) if h_i < a_i else
                                        ('head2_to_agent%i_weight' % (h_i+1), weg) for h_i, weg
                                        in enumerate(all_attend_probs[a_i][2][0].squeeze())),
                                   niter)
                logger.add_scalars('agent%i/attention' % a_i,
                                   dict(('to_agent%i_head_mean_weight' % h_i, mean_weight) if h_i < a_i else
                                        ('to_agent%i_head_mean_weight' % (h_i+1), mean_weight) for h_i, mean_weight
                                        in enumerate(torch.stack(all_attend_probs[a_i]).mean(0)[0].squeeze())), niter)
                logger.add_scalars('agent%i/attention' % a_i,
                                   dict(('head%i_entropy' % h_i, ent) for h_i, ent
                                        in enumerate(head_entropies)),
                                   niter)
            if len(agent_rets) == 1:
                all_rets.append(agent_rets[0])
            else:
                all_rets.append(agent_rets)
        if len(all_rets) == 1:
            return all_rets[0]
        else:
            return all_rets


class Critic(nn.Module):
    """
    Attention network, used as critic for all agents. Each agent gets its own
    observation and action, and can also attend over the other agents' encoded
    observations and actions.
    """
    def __init__(self, sa_sizes, hidden_dim=32, norm_in=True, attend_heads=1):
        """
        Inputs:
            sa_sizes (list of (int, int)): Size of state and action spaces per
                                          agent
            hidden_dim (int): Number of hidden dimensions
            norm_in (bool): Whether to apply BatchNorm to input
            attend_heads (int): Number of attention heads to use (use a number
                                that hidden_dim is divisible by)
        """
        super(Critic, self).__init__()
        self.sa_sizes = sa_sizes
        self.nagents = len(sa_sizes)

        self.critic_encoders = nn.ModuleList()
        self.critics = nn.ModuleList()

        self.state_encoders = nn.ModuleList()
        # iterate over agents
        for sdim, adim in sa_sizes:
            idim = sdim + adim
            odim = adim
            encoder = nn.Sequential()
            if norm_in:
                encoder.add_module('enc_bn', nn.BatchNorm1d(idim,
                                                            affine=False))
            encoder.add_module('enc_fc1', nn.Linear(idim, hidden_dim))
            encoder.add_module('enc_nl', nn.LeakyReLU())
            self.critic_encoders.append(encoder)
            critic = nn.Sequential()
            critic.add_module('critic_fc1', nn.Linear(2 * hidden_dim,
                                                      hidden_dim))
            critic.add_module('critic_nl', nn.LeakyReLU())
            critic.add_module('critic_fc2', nn.Linear(hidden_dim, odim))
            self.critics.append(critic)

            state_encoder = nn.Sequential()
            if norm_in:
                state_encoder.add_module('s_enc_bn', nn.BatchNorm1d(
                                            sdim, affine=False))
            state_encoder.add_module('s_enc_fc1', nn.Linear(sdim,
                                                            hidden_dim))
            state_encoder.add_module('s_enc_nl', nn.LeakyReLU())
            self.state_encoders.append(state_encoder)


    def forward(self, inps, act=None, agents=None, return_softmax_act=False, return_q=False, return_max_q=False,
                explore=False, return_all_q=False, logger=None, niter=0):
        """
        Inputs:
            inps (list of PyTorch Matrices): Inputs to each agents' encoder
                                             (batch of obs + ac)
            agents (int): indices of agents to return Q for
            return_q (bool): return Q-value
            return_all_q (bool): return Q-value for all actions
            regularize (bool): returns values to add to loss function for
                               regularization
            return_attend (bool): return attention weights per agent
            logger (TensorboardX SummaryWriter): If passed in, important values
                                                 are logged
        """
        if agents is None:
            agents = range(len(self.critic_encoders))
        states = [s for s, a in inps]    # Xq: obs.shape: nagents * bs * s_dim
        actions = [a for s, a in inps]   # Xq:    .shape: nagents * bs * a_dim
        if act is not None:
            actions = act
        inps = [torch.cat((s, a), dim=1) for s, a in inps]
        # extract state-action encoding for each agent
        sa_encodings = [encoder(inp) for encoder, inp in zip(self.critic_encoders, inps)] # Xq: sa_enc.shape: nagents * bs * h_dim
        # extract state encoding for each agent that we're returning Q for
        s_encodings = [self.state_encoders[a_i](states[a_i]) for a_i in agents]  # Xq: sa_enc.shape: nagents * bs * h_dim

        # gmf_L = (torch.stack(sa_encodings).permute(1, 2, 0)).mean(dim=2)
        gmf_Ls = []
        for i, a_i, selector in zip(range(len(agents)), agents, sa_encodings):
            keys = [k for j, k in enumerate(sa_encodings) if j != a_i]  # keys.shape: (nagents-1) * bs * attend_dim
            gmf_L = (torch.stack(keys).permute(1, 2, 0)).mean(dim=2)  # bs * attend_dim
            gmf_Ls.append(gmf_L)

        # calculate Q per agent
        all_rets = []
        for i, a_i in enumerate(agents):
            agent_rets = []
            critic_in = torch.cat((s_encodings[i], gmf_Ls[i]), dim=1)
            all_q = self.critics[a_i](critic_in)
            max_q = all_q.max(dim=1, keepdim=True)[0]
            int_acs = actions[a_i].max(dim=1, keepdim=True)[1]
            q = all_q.gather(1, int_acs)
            if return_softmax_act:
                probs = F.softmax(all_q, dim=1)
                on_gpu = next(self.parameters()).is_cuda
                if explore:
                    int_act, act = categorical_sample(probs, use_cuda=on_gpu)
                else:
                    act = onehot_from_logits(probs)
                agent_rets = [act]
                # if return_log_pi or return_entropy:
                #     log_probs = F.log_softmax(out, dim=1)
                # if return_all_probs:
                #     rets.append(probs)

            if return_q:
                agent_rets.append(q)
            if return_max_q:
                agent_rets.append(max_q)
            if return_all_q:
                agent_rets.append(all_q)
            if len(agent_rets) == 1:
                all_rets.append(agent_rets[0])
            else:
                all_rets.append(agent_rets)
        if len(all_rets) == 1:
            return all_rets[0]
        else:
            return all_rets


class AttentionIndepentCritic(nn.Module):
    """
    Attention network, used as critic for all agents. Each agent gets its own
    observation and action, and can also attend over the other agents' encoded
    observations and actions.
    """
    def __init__(self, sa_sizes, hidden_dim=32, norm_in=True, attend_heads=1):
        """
        Inputs:
            sa_sizes (list of (int, int)): Size of state and action spaces per
                                          agent
            hidden_dim (int): Number of hidden dimensions
            norm_in (bool): Whether to apply BatchNorm to input
            attend_heads (int): Number of attention heads to use (use a number
                                that hidden_dim is divisible by)
        """
        super(AttentionIndepentCritic, self).__init__()
        assert (hidden_dim % attend_heads) == 0
        self.sa_sizes = sa_sizes
        self.nagents = len(sa_sizes)
        self.attend_heads = attend_heads

        # self.critic_encoders = nn.ModuleList()
        # self.critics = nn.ModuleList()
        # self.state_encoders = nn.ModuleList()

        # iterate over agents
        sdim, adim = sa_sizes
        idim = sdim + adim
        odim = adim

        self.encoder = nn.Sequential()
        if norm_in:
            self.encoder.add_module('enc_bn', nn.BatchNorm1d(idim, affine=False))
        self.encoder.add_module('enc_fc1', nn.Linear(idim, hidden_dim))
        self.encoder.add_module('enc_nl', nn.LeakyReLU())
        #self.critic_encoders.append(encoder)

        self.critic = nn.Sequential()
        self.critic.add_module('critic_fc1', nn.Linear(2 * hidden_dim, hidden_dim))
        self.critic.add_module('critic_nl', nn.LeakyReLU())
        self.critic.add_module('critic_fc2', nn.Linear(hidden_dim, odim))
        #self.critics.append(critic)

        self.state_encoder = nn.Sequential()
        if norm_in:
            self.state_encoder.add_module('s_enc_bn', nn.BatchNorm1d(sdim, affine=False))
        self.state_encoder.add_module('s_enc_fc1', nn.Linear(sdim, hidden_dim))
        self.state_encoder.add_module('s_enc_nl', nn.LeakyReLU())
        #self.state_encoders.append(state_encoder)

        attend_dim = hidden_dim // attend_heads
        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()
        self.value_extractors = nn.ModuleList()
        for i in range(attend_heads):
            self.key_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.selector_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.value_extractors.append(nn.Sequential(nn.Linear(hidden_dim, attend_dim), nn.LeakyReLU()))
        self.shared_modules = [self.key_extractors, self.selector_extractors,
                               self.value_extractors, self.encoder]

    def shared_parameters(self):
        """
        Parameters shared across agents and reward heads
        """
        return chain(*[m.parameters() for m in self.shared_modules])

    def scale_shared_grads(self):
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        """
        for p in self.shared_parameters():
            p.grad.data.mul_(1. / self.nagents)


    def enc(self, state, action):
        # states = [s for s, a in inps]    # Xq: obs.shape: nagents * bs * s_dim
        # actions = [a for s, a in inps]   # Xq:    .shape: nagents * bs * a_dim
        # state, action = inps
        inp = torch.cat((state, action), dim=1)
        # extract state-action encoding for each agent
        sa_encoding = self.encoder(inp)  # Xq: sa_enc.shape: bs * h_dim
        # extract state encoding for each agent that we're returning Q for
        s_encoding = self.state_encoder(state)  # Xq: sa_enc.shape: bs * h_dim
        # extract keys for each head for each agent
        return [sa_encoding, s_encoding]

    def attend_cal(self, a_i, sa_encodings, s_encodings, agents=None):
        # if agents is None:key
        #     agents = range(len(self.critic_encoders))
        # all_head_keys = [[k_ext(enc) for enc in sa_encodings] for k_ext in self.key_extractors]
        all_head_keys = [[k_ext(enc) for enc in s_encodings] for k_ext in self.key_extractors]
        # Xq: all_head_keys.shape: heads * nagents * bs * attend_dim
        # extract sa values for each head for each agent
        all_head_values = [[v_ext(enc) for enc in sa_encodings] for v_ext in self.value_extractors]
        # extract selectors for each head for each agent that we're returning Q for
        all_head_selectors = [sel_ext(s_encodings[a_i])
                              for sel_ext in self.selector_extractors]
        # Xq: all_head_selectors.shape: heads * nagents * bs * attend_dim

        other_all_value = [ ]
        all_attend_logit = [ ]
        all_attend_prob = [ ]
        # calculate attention per head
        for curr_head_keys, curr_head_values, curr_head_selectors in zip(
                all_head_keys, all_head_values, all_head_selectors):
            # iterate over agents
            #for i, a_i, selector in zip(range(len(agents)), agents, curr_head_selectors):
            keys = [k for j, k in enumerate(curr_head_keys) if j != a_i]  # keys.shape: (nagents-1) * bs * attend_dim
            values = [v for j, v in enumerate(curr_head_values) if j != a_i]  # keys.shape: (nagents-1) * bs * attend_dim
            # calculate attention across agents
            # selector.shape: bs * attend_dim
            attend_logits = torch.matmul(curr_head_selectors.view(curr_head_selectors.shape[0], 1, -1),
                                         torch.stack(keys).permute(1, 2, 0))
            # attend_logits.shape: bs * 1 * (nagents-1) = torch.matmul( bs*1*attend_dim, bs*attend_dim*(nagents-1) )
            # scale dot-products by size of key (from Attention is All You Need)
            scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])
            attend_weights = F.softmax(scaled_attend_logits, dim=2)  # attend_weights.shape: bs * 1 * (nagents-1)
            other_values = (torch.stack(values).permute(1, 2, 0) *
                            attend_weights).sum(dim=2)   # other_values.shape: bs * attend_dim
            other_all_value.append(other_values)   # other_all_values.shape: nagents * bs * attend_dim
            all_attend_logit.append(attend_logits)
            all_attend_prob.append(attend_weights)
        return [other_all_value, all_attend_logit, all_attend_prob]

    def forward(self, a_i, act=None, agents=None, att=None, enc=None,
                return_softmax_act=False, return_q=False, return_trgt_q=False,
                return_max_q=False, regularize_pol=False,
                explore=False, return_all_q=False, regularize=False, return_log_pi=False, return_entropy=False,
                return_all_probs=False, return_attend=False, logger=None, niter=0):
        """
        Inputs:
            inps (list of PyTorch Matrices): Inputs to each agents' encoder
                                             (batch of obs + ac)
            agents (int): indices of agents to return Q for
            return_q (bool): return Q-value
            return_all_q (bool): return Q-value for all actions
            regularize (bool): returns values to add to loss function for
                               regularization
            return_attend (bool): return attention weights per agent
            logger (TensorboardX SummaryWriter): If passed in, important values
                                                 are logged
        """
        # if agents is None:
        #     agents = range(len(self.critic_encoders))
        # states = [s for s, a in inps]    # Xq: obs.shape: nagents * bs * s_dim
        # actions = [a for s, a in inps]   # Xq:    .shape: nagents * bs * a_dim
        if act is not None:
            actions = act
        # inps = [torch.cat((s, a), dim=1) for s, a in inps]
        # extract state-action encoding for each agent
        # sa_encodings = [encoder(inp) for encoder, inp in
        #                 zip(self.critic_encoders, inps)]  # Xq: sa_enc.shape: nagents * bs * h_dim
        # extract state encoding for each agent that we're returning Q for
        # s_encodings = [self.state_encoders[a_i](states[a_i]) for a_i in
        #                agents]  # Xq: sa_enc.shape: nagents * bs * h_dim

        # inp = torch.cat((states, actions), dim=1)
        # # extract state-action encoding for each agent
        # sa_encoding = self.encoder(inp) # Xq: sa_enc.shape: bs * h_dim
        # # extract state encoding for each agent that we're returning Q for
        # s_encoding = self.state_encoder(states)  # Xq: sa_enc.shape: bs * h_dim
        # states = [s for s, a in inps]  # Xq: obs.shape: nagents * bs * s_dim
        # actions = [a for s, a in inps]  # Xq:    .shape: nagents * bs * a_dim

        # # extract state-action encoding for each agent
        # sa_encodings = [encoder(inp) for encoder, inp in
        #                 zip(self.critic_encoders, inps)]  # Xq: sa_enc.shape: nagents * bs * h_dim
        # # extract state encoding for each agent that we're returning Q for
        # s_encodings = [self.state_encoders[a_i](states[a_i]) for a_i in
        #                agents]  # Xq: sa_enc.shape: nagents * bs * h_dim

        # extract keys for each head for each agent
        # all_head_keys = [[k_ext(enc) for enc in sa_encodings] for k_ext in self.key_extractors]
        # # Xq: all_head_keys.shape: heads * nagents * bs * attend_dim
        # # extract sa values for each head for each agent
        # all_head_values = [[v_ext(enc) for enc in sa_encodings] for v_ext in self.value_extractors]
        # # extract selectors for each head for each agent that we're returning Q for
        # all_head_selectors = [[sel_ext(enc) for i, enc in enumerate(s_encodings) if i in agents]
        #                       for sel_ext in self.selector_extractors]
        # # Xq: all_head_selectors.shape: heads * nagents * bs * attend_dim
        #
        # other_all_values = [[] for _ in range(len(agents))]
        # all_attend_logits = [[] for _ in range(len(agents))]
        # all_attend_probs = [[] for _ in range(len(agents))]
        # # calculate attention per head
        # for curr_head_keys, curr_head_values, curr_head_selectors in zip(
        #         all_head_keys, all_head_values, all_head_selectors):
        #     # iterate over agents
        #     for i, a_i, selector in zip(range(len(agents)), agents, curr_head_selectors):
        #         keys = [k for j, k in enumerate(curr_head_keys) if
        #                 j != a_i]  # keys.shape: (nagents-1) * bs * attend_dim
        #         values = [v for j, v in enumerate(curr_head_values) if
        #                   j != a_i]  # keys.shape: (nagents-1) * bs * attend_dim
        #         # calculate attention across agents
        #         # selector.shape: bs * attend_dim
        #         attend_logits = torch.matmul(selector.view(selector.shape[0], 1, -1),
        #                                      torch.stack(keys).permute(1, 2, 0))
        #         # attend_logits.shape: bs * 1 * (nagents-1) = torch.matmul( bs*1*attend_dim, bs*attend_dim*(nagents-1) )
        #         # scale dot-products by size of key (from Attention is All You Need)
        #         scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])
        #         attend_weights = F.softmax(scaled_attend_logits, dim=2)  # attend_weights.shape: bs * 1 * (nagents-1)
        #         other_values = (torch.stack(values).permute(1, 2, 0) *
        #                         attend_weights).sum(dim=2)  # other_values.shape: bs * attend_dim
        #         other_all_values[i].append(other_values)  # other_all_values.shape: nagents * bs * attend_dim
        #         all_attend_logits[i].append(attend_logits)
        #         all_attend_probs[i].append(attend_weights)
        # calculate Q per agent
        other_all_values, all_attend_logits, all_attend_probs = att
        sa_encodings, s_encodings = enc

        head_entropies = [(-((probs + 1e-8).log() * probs).squeeze().sum(1)
                           .mean()) for probs in all_attend_probs[a_i]]
        agent_rets = []
        critic_in = torch.cat((s_encodings[a_i], *other_all_values[a_i]), dim=1)
        all_q = self.critic(critic_in)
        max_q = all_q.max(dim=1, keepdim=True)[0]
        int_acs = actions[a_i].max(dim=1, keepdim=True)[1]
        q = all_q.gather(1, int_acs)
        if return_softmax_act:
            probs = F.softmax(all_q, dim=1)
            on_gpu = next(self.parameters()).is_cuda
            if explore:
                int_act, act = categorical_sample(probs, use_cuda=on_gpu)
            else:
                act = onehot_from_logits(probs)
            agent_rets = [act]
        if return_log_pi or return_entropy:
            log_probs = F.log_softmax(all_q, dim=1)
        if return_log_pi:
            # return log probability of selected action
            agent_rets.append(log_probs.gather(1, int_act))
        if return_entropy:
            agent_rets.append(-(log_probs * probs).sum(1).mean())
        if return_all_probs:
            agent_rets.append(probs)
        if return_q:
            agent_rets.append(q)
        if return_trgt_q:
            agent_rets.append(all_q.gather(1, int_acs))
        if return_max_q:
            agent_rets.append(max_q)
        if return_all_q:
            agent_rets.append(all_q)
        if regularize:
            # regularize magnitude of attention logits
            attend_mag_reg = 1e-3 * sum((logit**2).mean() for logit in
                                        all_attend_logits[a_i])
            regs = (attend_mag_reg,)
            agent_rets.append(regs)
        if regularize_pol:
            agent_rets.append(1e-3 * (all_q ** 2).mean())
        if return_attend:
            agent_rets.append(np.array(all_attend_probs[a_i]))
        if logger is not None:
            # all_attend_probs[a_i].shape  head_num * bs * 1 * nagents-1
            logger.add_scalars('agent%i/attention' % a_i,
                               dict(('head0_to_agent%i_weight' % h_i, weg) if h_i < a_i else
                                    ('head0_to_agent%i_weight' % (h_i+1), weg) for h_i, weg
                                    in enumerate(all_attend_probs[a_i][0][0].squeeze())),
                               niter)
            logger.add_scalars('agent%i/attention' % a_i,
                               dict(('head2_to_agent%i_weight' % h_i, weg) if h_i < a_i else
                                    ('head2_to_agent%i_weight' % (h_i+1), weg) for h_i, weg
                                    in enumerate(all_attend_probs[a_i][2][0].squeeze())),
                               niter)
            logger.add_scalars('agent%i/attention' % a_i,
                               dict(('to_agent%i_head_mean_weight' % h_i, mean_weight) if h_i < a_i else
                                    ('to_agent%i_head_mean_weight' % (h_i+1), mean_weight) for h_i, mean_weight
                                    in enumerate(torch.stack(all_attend_probs[a_i]).mean(0)[0].squeeze())), niter)
            logger.add_scalars('agent%i/attention' % a_i,
                               dict(('head%i_entropy' % h_i, ent) for h_i, ent
                                    in enumerate(head_entropies)),
                               niter)
        if len(agent_rets) == 1:
            return agent_rets[0]
        else:
            return agent_rets

"""
Policy functions for training and update
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.utils import batch_to_seq, init_layer, one_hot, run_rnn


class Policy(nn.Module):
    def __init__(self, n_a, n_s, n_step, policy_name, agent_name, identical, device=None):
        super(Policy, self).__init__()
        self.name = policy_name
        if agent_name is not None:
            # for multi-agent system
            self.name += '_' + str(agent_name)
        self.n_a = n_a
        self.n_s = n_s
        self.n_step = n_step
        self.identical = identical
        self.device = device

    def forward(self, ob, *_args, **_kwargs):
        raise NotImplementedError()

    def _init_actor_head(self, n_h, n_a=None):
        if n_a is None:
            n_a = self.n_a
        # only discrete control is supported for now
        self.actor_head = nn.Linear(n_h, n_a)
        init_layer(self.actor_head, 'fc')

    def _init_critic_head(self, n_h, n_n=None):
        if n_n is None:
            n_n = int(self.n_n)
        if n_n:
            if self.identical:
                n_na_sparse = self.n_a * n_n
            else:
                n_na_sparse = sum(self.na_dim_ls)
            n_h += n_na_sparse
        self.critic_head = nn.Linear(n_h, 1)
        init_layer(self.critic_head, 'fc')

    def _run_critic_head(self, h, na, n_n=None):
        if n_n is None:
            n_n = int(self.n_n)
        if n_n:
            na = torch.from_numpy(na).long().to(self.device)
            if self.identical:
                na_sparse = one_hot(na, self.n_a)
                na_sparse = na_sparse.view(-1, self.n_a * n_n)
            else:
                na_sparse = []
                na_ls = torch.chunk(na, n_n, dim=1)
                for na_val, na_dim in zip(na_ls, self.na_dim_ls):
                    na_sparse.append(torch.squeeze(one_hot(na_val, na_dim), dim=1))
                na_sparse = torch.cat(na_sparse, dim=1)
            h = torch.cat([h, na_sparse], dim=1)
        return self.critic_head(h).squeeze()

    def _run_loss(self, actor_dist, e_coef, v_coef, vs, As, Rs, Advs):
        log_probs = actor_dist.log_prob(As)
        policy_loss = -(log_probs * Advs).mean()
        entropy_loss = -(actor_dist.entropy()).mean() * e_coef
        value_loss = (Rs - vs).pow(2).mean() * v_coef
        return policy_loss, value_loss, entropy_loss

    def _update_tensorboard(self, summary_writer, global_step):
        # monitor training
        summary_writer.add_scalar('loss/entropy_loss', self.entropy_loss,
                                  global_step=global_step)
        summary_writer.add_scalar('loss/policy_loss', self.policy_loss,
                                  global_step=global_step)
        summary_writer.add_scalar('loss/value_loss', self.value_loss,
                                  global_step=global_step)
        summary_writer.add_scalar('loss/total_loss', self.loss,
                                  global_step=global_step)


class LstmPolicy(Policy):
    def __init__(self, n_s, n_a, n_n, n_step, n_fc=64, n_lstm=64, name=None,
                 na_dim_ls=None, identical=True):
        super(LstmPolicy, self).__init__(n_a, n_s, n_step, 'lstm', name, identical)
        if not self.identical:
            self.na_dim_ls = na_dim_ls
        self.n_lstm = n_lstm
        self.n_fc = n_fc
        self.n_n = n_n
        self._init_net()
        self._reset()

    def backward(self, obs, nactions, acts, dones, Rs, Advs,
                 e_coef, v_coef, summary_writer=None, global_step=None):
        obs = torch.from_numpy(obs).float()
        dones = torch.from_numpy(dones).float()
        xs = self._encode_ob(obs)
        hs, new_states = run_rnn(self.lstm_layer, xs, dones, self.states_bw)
        # backward grad is limited to the minibatch
        self.states_bw = new_states.detach()
        actor_dist = torch.distributions.categorical.Categorical(logits=F.log_softmax(self.actor_head(hs), dim=1))
        vs = self._run_critic_head(hs, nactions)
        self.policy_loss, self.value_loss, self.entropy_loss = \
            self._run_loss(actor_dist, e_coef, v_coef, vs,
                           torch.from_numpy(acts).long(),
                           torch.from_numpy(Rs).float(),
                           torch.from_numpy(Advs).float())
        self.loss = self.policy_loss + self.value_loss + self.entropy_loss
        self.loss.backward()
        if summary_writer is not None:
            self._update_tensorboard(summary_writer, global_step)

    def forward(self, ob, done, naction=None, out_type='p'):
        ob = torch.from_numpy(np.expand_dims(ob, axis=0)).float()
        done = torch.from_numpy(np.expand_dims(done, axis=0)).float()
        x = self._encode_ob(ob)
        h, new_states = run_rnn(self.lstm_layer, x, done, self.states_fw)
        if out_type.startswith('p'):
            self.states_fw = new_states.detach()
            return F.softmax(self.actor_head(h), dim=1).squeeze().detach().numpy()
        else:
            return self._run_critic_head(h, np.array([naction])).detach().numpy()

    def _encode_ob(self, ob):
        return F.relu(self.fc_layer(ob))

    def _init_net(self):
        self.fc_layer = nn.Linear(self.n_s, self.n_fc)
        init_layer(self.fc_layer, 'fc')
        self.lstm_layer = nn.LSTMCell(self.n_fc, self.n_lstm)
        init_layer(self.lstm_layer, 'lstm')
        self._init_actor_head(self.n_lstm)
        self._init_critic_head(self.n_lstm)

    def _reset(self):
        # 重置每个 cum_step 的累积状态
        self.states_fw = torch.zeros(self.n_lstm * 2)
        self.states_bw = torch.zeros(self.n_lstm * 2)


class FPPolicy(LstmPolicy):
    def __init__(self, n_s, n_a, n_n, n_step, n_fc=64, n_lstm=64, name=None,
                 na_dim_ls=None, identical=True):
        super(FPPolicy, self).__init__(n_s, n_a, n_n, n_step, n_fc, n_lstm, name,
                                       na_dim_ls, identical)

    def _init_net(self):
        if self.identical:
            # dim of obs except the fingerprints
            self.n_x = self.n_s - self.n_n * self.n_a
        else:
            self.n_x = int(self.n_s - sum(self.na_dim_ls))
        self.fc_x_layer = nn.Linear(self.n_x, self.n_fc)
        init_layer(self.fc_x_layer, 'fc')
        n_h = self.n_fc
        if self.n_n:
            self.fc_p_layer = nn.Linear(self.n_s - self.n_x, self.n_fc)
            init_layer(self.fc_p_layer, 'fc')
            n_h += self.n_fc
        self.lstm_layer = nn.LSTMCell(n_h, self.n_lstm)
        init_layer(self.lstm_layer, 'lstm')
        self._init_actor_head(self.n_lstm)
        self._init_critic_head(self.n_lstm)

    def _encode_ob(self, ob):
        x = F.relu(self.fc_x_layer(ob[:, :self.n_x]))
        if self.n_n:
            p = F.relu(self.fc_p_layer(ob[:, self.n_x:]))
            x = torch.cat([x, p], dim=1)
        return x


class NCMultiAgentPolicy(Policy):
    """ Implemented as a centralized meta-DNN. To simplify the implementation, all input
    and output dimensions are identical among all agents, and invalid values are casted as
    zeros during runtime."""

    def __init__(self, n_s, n_a, n_agent, n_step, neighbor_mask, n_fc=64, n_h=64,
                 n_s_ls=None, n_a_ls=None, identical=True):
        """
        n_h: dim of hidden states
        """
        super(NCMultiAgentPolicy, self).__init__(n_a, n_s, n_step, 'nc', None, identical)
        if not self.identical:
            self.n_s_ls = n_s_ls
            self.n_a_ls = n_a_ls
        self.n_agent = n_agent
        self.neighbor_mask = neighbor_mask
        self.n_fc = n_fc
        self.n_h = n_h
        self._init_net()
        self._reset()

    def _get_neighbor_dim(self, i_agent):
        # return num of neighbors, dim of overall states, sum of neigbor's action dim,
        # list of neighbor-information's dim, list of neighbor-action's dim
        n_n = int(np.sum(self.neighbor_mask[i_agent]))
        if self.identical:
            return n_n, self.n_s * (n_n + 1), self.n_a * n_n, [self.n_s] * n_n, [self.n_a] * n_n
        else:
            ns_ls = []
            na_ls = []
            for j in np.where(self.neighbor_mask[i_agent])[0]:
                ns_ls.append(self.n_s_ls[j])
                na_ls.append(self.n_a_ls[j])
            return n_n, self.n_s_ls[i_agent] + sum(ns_ls), sum(na_ls), ns_ls, na_ls

    def _init_actor_head(self, n_a):
        # only discrete control is supported for now
        actor_head = nn.Linear(self.n_h, n_a)
        init_layer(actor_head, 'fc')
        self.actor_heads.append(actor_head)

    def _init_critic_head(self, n_na):
        # TODO: can we include more information,like neighbor's hidden states
        critic_head = nn.Linear(self.n_h + n_na, 1)
        init_layer(critic_head, 'fc')
        self.critic_heads.append(critic_head)

    def _init_net(self):
        self.fc_x_layers = nn.ModuleList()
        self.fc_p_layers = nn.ModuleList()
        self.fc_m_layers = nn.ModuleList()
        self.lstm_layers = nn.ModuleList()
        self.actor_heads = nn.ModuleList()
        self.critic_heads = nn.ModuleList()
        self.ns_ls_ls = []
        self.na_ls_ls = []
        self.n_n_ls = []
        for i in range(self.n_agent):
            n_n, n_ns, n_na, ns_ls, na_ls = self._get_neighbor_dim(i)
            self.ns_ls_ls.append(ns_ls)
            self.na_ls_ls.append(na_ls)
            self.n_n_ls.append(n_n)
            self._init_comm_layer(n_n, n_ns, n_na)
            n_a = self.n_a if self.identical else self.n_a_ls[i]
            self._init_actor_head(n_a)
            self._init_critic_head(n_na)

    def _reset(self):
        self.states_fw = torch.zeros(self.n_agent, self.n_h * 2).to(self.device)
        self.states_bw = torch.zeros(self.n_agent, self.n_h * 2).to(self.device)

    def _run_actor_heads(self, hs, detach=False):
        ps = []
        for i in range(self.n_agent):
            if detach:
                p_i = F.softmax(self.actor_heads[i](hs[i]), dim=1).squeeze().detach().cpu().numpy()
            else:
                p_i = F.log_softmax(self.actor_heads[i](hs[i]), dim=1)
            ps.append(p_i)
        return ps

    def _run_critic_heads(self, hs, actions, detach=False):
        # include neighbor's actions (one-hot) to get the value fn
        vs = []
        for i in range(self.n_agent):
            n_n = self.n_n_ls[i]
            if n_n:
                js = torch.from_numpy(np.where(self.neighbor_mask[i])[0]).long().to(self.device)
                na_i = torch.index_select(actions, 0, js)
                na_i_ls = []
                for j in range(n_n):
                    na_i_ls.append(one_hot(na_i[j], self.na_ls_ls[i][j]))
                h_i = torch.cat([hs[i]] + na_i_ls, dim=1)
            else:
                h_i = hs[i]
            v_i = self.critic_heads[i](h_i).squeeze()
            if detach:
                vs.append(v_i.detach().cpu().numpy())
            else:
                vs.append(v_i)
        return vs

    def backward(self, obs, fps, acts, dones, Rs, Advs,
                 e_coef, v_coef, summary_writer=None, global_step=None):
        obs = torch.from_numpy(obs).float().transpose(0, 1).to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)
        fps = torch.from_numpy(fps).float().transpose(0, 1).to(self.device)
        acts = torch.from_numpy(acts).long().to(self.device)
        hs, new_states = self._run_comm_layers(obs, dones, fps, self.states_bw)
        # backward grad is limited to the minibatch
        self.states_bw = new_states.detach()
        ps = self._run_actor_heads(hs)
        vs = self._run_critic_heads(hs, acts)
        self.policy_loss = 0
        self.value_loss = 0
        self.entropy_loss = 0
        Rs = torch.from_numpy(Rs).float().to(self.device)
        Advs = torch.from_numpy(Advs).float().to(self.device)
        for i in range(self.n_agent):
            actor_dist_i = torch.distributions.categorical.Categorical(logits=ps[i])
            policy_loss_i, value_loss_i, entropy_loss_i = \
                self._run_loss(actor_dist_i, e_coef, v_coef, vs[i],
                               acts[i], Rs[i], Advs[i])
            self.policy_loss += policy_loss_i
            self.value_loss += value_loss_i
            self.entropy_loss += entropy_loss_i
        self.loss = self.policy_loss + self.value_loss + self.entropy_loss
        self.loss.backward()
        if summary_writer is not None:
            self._update_tensorboard(summary_writer, global_step)

    def forward(self, ob, done=None, fp=None, action=None, out_type='p'):
        # TxNxm
        ob = torch.from_numpy(np.expand_dims(ob, axis=0)).float().to(self.device)
        done = torch.from_numpy(np.expand_dims(done, axis=0)).float().to(self.device)
        fp = torch.from_numpy(np.expand_dims(fp, axis=0)).float().to(self.device)
        # h dim: NxTxm
        h, new_states = self._run_comm_layers(ob, done, fp, self.states_fw)
        if out_type.startswith('p'):
            self.states_fw = new_states.detach()
            return self._run_actor_heads(h, detach=True)
        else:
            action = torch.from_numpy(np.expand_dims(action, axis=1)).long().to(self.device)
            return self._run_critic_heads(h, action, detach=True)

    def _init_comm_layer(self, n_n, n_ns, n_na):
        # num of neighbors, neighbor-information's dim, neighbor-policy's dim
        n_lstm_in = 3 * self.n_fc
        fc_x_layer = nn.Linear(n_ns, self.n_fc)
        init_layer(fc_x_layer, 'fc')
        self.fc_x_layers.append(fc_x_layer)
        if n_n:
            fc_p_layer = nn.Linear(n_na, self.n_fc)
            init_layer(fc_p_layer, 'fc')
            fc_m_layer = nn.Linear(self.n_h * n_n, self.n_fc)
            init_layer(fc_m_layer, 'fc')
            self.fc_m_layers.append(fc_m_layer)
            self.fc_p_layers.append(fc_p_layer)
            lstm_layer = nn.LSTMCell(n_lstm_in, self.n_h)
        else:
            self.fc_m_layers.append(None)
            self.fc_p_layers.append(None)
            lstm_layer = nn.LSTMCell(self.n_fc, self.n_h)
        init_layer(lstm_layer, 'lstm')
        self.lstm_layers.append(lstm_layer)

    def _run_comm_layers(self, obs, dones, fps, states):
        # encode comm info(s_i) and (h_(t-1), c_(t-1)) to the h_t and c_t
        # states (h_(t-1), c_(t-1)): t-1 hidden states, (self.n_agent, self.n_h * 2)
        obs = batch_to_seq(obs)
        dones = batch_to_seq(dones)
        fps = batch_to_seq(fps)
        h, c = torch.chunk(states, 2, dim=1)
        outputs = []
        for t, (x, p, done) in enumerate(zip(obs, fps, dones)):
            next_h = []
            next_c = []
            x = x.squeeze(0)
            p = p.squeeze(0)
            for i in range(self.n_agent):
                n_n = self.n_n_ls[i]
                if n_n:
                    s_i = self._get_comm_s(i, n_n, x, h, p)
                else:
                    s_i = F.relu(self.fc_x_layers[i](x[i].unsqueeze(0)))
                h_i, c_i = h[i].unsqueeze(0) * (1 - done), c[i].unsqueeze(0) * (1 - done)
                # print('self.device', self.device)
                # print('self.lstm_layers[i]', type(self.lstm_layers[i]))
                # print('self.lstm_layers[i]', next(self.lstm_layers[i].parameters()).device)
                # print('self.lstm_layers[i]', type(s_i), s_i.device)
                # print('self.lstm_layers[i]', type(h_i), h_i.device)
                # print('self.lstm_layers[i]', type(c_i), c_i.device)
                # h_i, c_i = h_i.to(self.device), c_i.to(self.device)
                # print(torch.cuda.get_device_properties(0).total_memory)
                # h_i, c_i = h_i.cuda(), c_i.cuda()
                next_h_i, next_c_i = self.lstm_layers[i](s_i, (h_i, c_i))
                next_h.append(next_h_i)
                next_c.append(next_c_i)
            h, c = torch.cat(next_h), torch.cat(next_c)
            outputs.append(h.unsqueeze(0))
        outputs = torch.cat(outputs)
        return outputs.transpose(0, 1), torch.cat([h, c], dim=1)

    def _get_comm_s(self, i, n_n, x, h, p):
        # get the S_(i-1), h_(i-1), Pi_(i-1)
        js = torch.from_numpy(np.where(self.neighbor_mask[i])[0]).long().to(self.device)
        m_i = torch.index_select(h, 0, js).view(1, self.n_h * n_n)  # neighbor's hidden states
        p_i = torch.index_select(p, 0, js)  # neighbor's policy
        nx_i = torch.index_select(x, 0, js)  # neighbor's obs
        if self.identical:
            p_i = p_i.view(1, self.n_a * n_n)
            nx_i = nx_i.view(1, self.n_s * n_n)
        else:
            p_i_ls = []
            nx_i_ls = []
            for j in range(n_n):
                p_i_ls.append(p_i[j].narrow(0, 0, self.na_ls_ls[i][j]))
                nx_i_ls.append(nx_i[j].narrow(0, 0, self.ns_ls_ls[i][j]))
            p_i = torch.cat(p_i_ls).unsqueeze(0)
            nx_i = torch.cat(nx_i_ls).unsqueeze(0)

        # encode neighbor's states along with its own states;
        # encode neighbor's policies;
        # encode neighbor's hidden states.
        s_i = [F.relu(self.fc_x_layers[i](torch.cat([x[i].unsqueeze(0), nx_i], dim=1))),
               F.relu(self.fc_p_layers[i](p_i)),
               F.relu(self.fc_m_layers[i](m_i))]
        return torch.cat(s_i, dim=1)


class PowerNetMultiAgentPolicy(NCMultiAgentPolicy):
    """PowerNet"""

    def __init__(self, n_s, n_a, n_agent, n_step, neighbor_mask, n_fc=64, n_h=64,
                 n_s_ls=None, n_a_ls=None, identical=True, device=None):
        Policy.__init__(self, n_a, n_s, n_step, 'pnet', None, identical, device)
        if not self.identical:
            self.n_s_ls = n_s_ls
            self.n_a_ls = n_a_ls
        self.n_agent = n_agent
        self.neighbor_mask = neighbor_mask
        self.n_fc = n_fc
        self.n_h = n_h
        self._init_net()
        self._reset()

    def _init_net(self):
        self.fc_x_layers = nn.ModuleList()
        self.fc_x1_layers = nn.ModuleList()
        self.fc_x2_layers = nn.ModuleList()
        self.fc_x3_layers = nn.ModuleList()
        self.fc_x4_layers = nn.ModuleList()
        self.fc_p_layers = nn.ModuleList()
        self.fc_m_layers = nn.ModuleList()
        self.lstm_layers = nn.ModuleList()
        self.actor_heads = nn.ModuleList()
        self.critic_heads = nn.ModuleList()
        self.ns_ls_ls = []
        self.na_ls_ls = []
        self.n_n_ls = []
        for i in range(self.n_agent):
            n_n, n_ns, n_na, ns_ls, na_ls = self._get_neighbor_dim(i)
            self.ns_ls_ls.append(ns_ls)
            self.na_ls_ls.append(na_ls)
            self.n_n_ls.append(n_n)
            self._init_comm_layer(n_n, n_ns, n_na)
            n_a = self.n_a if self.identical else self.n_a_ls[i]
            self._init_actor_head(n_a)
            self._init_critic_head(n_na)

    def _init_comm_layer(self, n_n, n_ns, n_na):
        n_lstm_in = 2 * self.n_fc
        # fc_x_layer = nn.Linear(n_ns, self.n_fc)
        fc_x_layer = nn.Linear(self.n_fc, self.n_fc)
        init_layer(fc_x_layer, 'fc')
        self.fc_x_layers.append(fc_x_layer)

        fc_x1_layer = nn.Linear(self.n_s // 9, 8)
        init_layer(fc_x1_layer, 'fc')
        self.fc_x1_layers.append(fc_x1_layer)

        fc_x2_layer = nn.Linear(self.n_s // 9 * 2, 16)
        init_layer(fc_x2_layer, 'fc')
        self.fc_x2_layers.append(fc_x2_layer)

        fc_x3_layer = nn.Linear(self.n_s // 9 * 4, 24)
        init_layer(fc_x3_layer, 'fc')
        self.fc_x3_layers.append(fc_x3_layer)

        fc_x4_layer = nn.Linear(self.n_s // 9 * 2, 16)
        init_layer(fc_x4_layer, 'fc')
        self.fc_x4_layers.append(fc_x4_layer)
        init_layer(fc_x_layer, 'fc')
        self.fc_x_layers.append(fc_x_layer)
        if n_n:
            fc_m_layer = nn.Linear(self.n_h, self.n_fc)
            init_layer(fc_m_layer, 'fc')
            self.fc_m_layers.append(fc_m_layer)
            lstm_layer = nn.LSTMCell(n_lstm_in, self.n_h)
        else:
            self.fc_m_layers.append(None)
            lstm_layer = nn.LSTMCell(self.n_fc, self.n_h)
        init_layer(lstm_layer, 'lstm')
        self.lstm_layers.append(lstm_layer)

    def _get_comm_s(self, i, n_n, x, h, p):
        # we only pass hidden states as comm info
        js = torch.from_numpy(np.where(self.neighbor_mask[i])[0]).long().to(self.device)  # tensor([1, 2])
        m_i = torch.index_select(h, 0, js).mean(dim=0, keepdim=True)
        # print(type(m_i))
        if self.identical:
            x_i = x[i].unsqueeze(0)
            mx1_i = self.fc_x1_layers[i](x_i[:, 0].unsqueeze(0))
            mx2_i = self.fc_x2_layers[i](torch.flatten(x_i[:, 1:3]).unsqueeze(0))
            mx3_i = self.fc_x3_layers[i](torch.flatten(x_i[:, 3:7]).unsqueeze(0))
            mx4_i = self.fc_x4_layers[i](torch.flatten(x_i[:, 7:]).unsqueeze(0))
            mx_i = torch.cat([mx1_i, mx2_i, mx3_i, mx4_i], dim=1)
            # print(type(mx_i))
        s_i = [F.relu(self.fc_x_layers[i](mx_i)), self.fc_m_layers[i](m_i)]
        # print('s_i', type(s_i[0]), type(s_i[1]), s_i[0].shape, s_i[1].shape, s_i[0].device, s_i[1].device)
        return torch.cat(s_i, dim=1)


class ConsensusPolicy(NCMultiAgentPolicy):
    def __init__(self, n_s, n_a, n_agent, n_step, neighbor_mask, n_fc=64, n_h=64,
                 n_s_ls=None, n_a_ls=None, identical=True):
        Policy.__init__(self, n_a, n_s, n_step, 'cu', None, identical)
        if not self.identical:
            self.n_s_ls = n_s_ls
            self.n_a_ls = n_a_ls
        self.n_agent = n_agent
        self.neighbor_mask = neighbor_mask
        self.n_fc = n_fc
        self.n_h = n_h
        self._init_net()
        self._reset()

    def consensus_update(self):
        consensus_update = []
        with torch.no_grad():
            for i in range(self.n_agent):
                mean_wts = self._get_critic_wts(i)
                for param, wt in zip(self.lstm_layers[i].parameters(), mean_wts):
                    param.copy_(wt)

    def _init_net(self):
        self.fc_x_layers = nn.ModuleList()
        self.lstm_layers = nn.ModuleList()
        self.actor_heads = nn.ModuleList()
        self.critic_heads = nn.ModuleList()
        self.na_ls_ls = []
        self.n_n_ls = []
        for i in range(self.n_agent):
            n_n, _, n_na, _, na_ls = self._get_neighbor_dim(i)
            n_s = self.n_s if self.identical else self.n_s_ls[i]
            self.na_ls_ls.append(na_ls)
            self.n_n_ls.append(n_n)
            fc_x_layer = nn.Linear(n_s, self.n_fc)
            init_layer(fc_x_layer, 'fc')
            self.fc_x_layers.append(fc_x_layer)
            lstm_layer = nn.LSTMCell(self.n_fc, self.n_h)
            init_layer(lstm_layer, 'lstm')
            self.lstm_layers.append(lstm_layer)
            n_a = self.n_a if self.identical else self.n_a_ls[i]
            self._init_actor_head(n_a)
            self._init_critic_head(n_na)

    def _get_critic_wts(self, i_agent):
        wts = []
        for wt in self.lstm_layers[i_agent].parameters():
            wts.append(wt.detach())
        neighbors = list(np.where(self.neighbor_mask[i_agent] == 1)[0])
        for j in neighbors:
            for k, wt in enumerate(self.lstm_layers[j].parameters()):
                wts[k] += wt.detach()
        n = 1 + len(neighbors)
        for k in range(len(wts)):
            wts[k] /= n
        return wts

    def _run_comm_layers(self, obs, dones, fps, states):
        # NxTxm
        obs = obs.transpose(0, 1)
        hs = []
        new_states = []
        for i in range(self.n_agent):
            xs_i = F.relu(self.fc_x_layers[i](obs[i]))
            hs_i, new_states_i = run_rnn(self.lstm_layers[i], xs_i, dones, states[i])
            hs.append(hs_i.unsqueeze(0))
            new_states.append(new_states_i.unsqueeze(0))
        return torch.cat(hs), torch.cat(new_states)


class CommNetMultiAgentPolicy(NCMultiAgentPolicy):
    """Reference code: https://github.com/IC3Net/IC3Net/blob/master/comm.py.
       Note in CommNet, the message is generated from hidden state only, so current state
       and neigbor policies are not included in the inputs."""

    def __init__(self, n_s, n_a, n_agent, n_step, neighbor_mask, n_fc=64, n_h=64,
                 n_s_ls=None, n_a_ls=None, identical=True):
        Policy.__init__(self, n_a, n_s, n_step, 'cnet', None, identical)
        if not self.identical:
            self.n_s_ls = n_s_ls
            self.n_a_ls = n_a_ls
        self.n_agent = n_agent
        self.neighbor_mask = neighbor_mask
        self.n_fc = n_fc
        self.n_h = n_h
        self._init_net()
        self._reset()

    def _init_comm_layer(self, n_n, n_ns, n_na):
        fc_x_layer = nn.Linear(n_ns, self.n_fc)
        init_layer(fc_x_layer, 'fc')
        self.fc_x_layers.append(fc_x_layer)
        if n_n:
            fc_m_layer = nn.Linear(self.n_h, self.n_fc)
            init_layer(fc_m_layer, 'fc')
            self.fc_m_layers.append(fc_m_layer)
        else:
            self.fc_m_layers.append(None)
        lstm_layer = nn.LSTMCell(self.n_fc, self.n_h)
        init_layer(lstm_layer, 'lstm')
        self.lstm_layers.append(lstm_layer)

    def _get_comm_s(self, i, n_n, x, h, p):
        js = torch.from_numpy(np.where(self.neighbor_mask[i])[0]).long()
        m_i = torch.index_select(h, 0, js).mean(dim=0, keepdim=True)
        nx_i = torch.index_select(x, 0, js)
        if self.identical:
            nx_i = nx_i.view(1, self.n_s * n_n)
        else:
            nx_i_ls = []
            for j in range(n_n):
                nx_i_ls.append(nx_i[j].narrow(0, 0, self.ns_ls_ls[i][j]))
            nx_i = torch.cat(nx_i_ls).unsqueeze(0)
        return F.relu(self.fc_x_layers[i](torch.cat([x[i].unsqueeze(0), nx_i], dim=1))) + \
               self.fc_m_layers[i](m_i)


class DIALMultiAgentPolicy(NCMultiAgentPolicy):
    def __init__(self, n_s, n_a, n_agent, n_step, neighbor_mask, n_fc=64, n_h=64,
                 n_s_ls=None, n_a_ls=None, identical=True):
        Policy.__init__(self, n_a, n_s, n_step, 'dial', None, identical)
        if not self.identical:
            self.n_s_ls = n_s_ls
            self.n_a_ls = n_a_ls
        self.n_agent = n_agent
        self.neighbor_mask = neighbor_mask
        self.n_fc = n_fc
        self.n_h = n_h
        self._init_net()
        self._reset()

    def _init_comm_layer(self, n_n, n_ns, n_na):
        fc_x_layer = nn.Linear(n_ns, self.n_fc)
        init_layer(fc_x_layer, 'fc')
        self.fc_x_layers.append(fc_x_layer)
        if n_n:
            fc_m_layer = nn.Linear(self.n_h * n_n, self.n_fc)
            init_layer(fc_m_layer, 'fc')
            self.fc_m_layers.append(fc_m_layer)
        else:
            self.fc_m_layers.append(None)
        lstm_layer = nn.LSTMCell(self.n_fc, self.n_h)
        init_layer(lstm_layer, 'lstm')
        self.lstm_layers.append(lstm_layer)

    def _get_comm_s(self, i, n_n, x, h, p):
        # different with NC, include only it's own action, no other's policies.
        # summation all the info
        js = torch.from_numpy(np.where(self.neighbor_mask[i])[0]).long()
        m_i = torch.index_select(h, 0, js).view(1, self.n_h * n_n)
        nx_i = torch.index_select(x, 0, js)
        if self.identical:
            nx_i = nx_i.view(1, self.n_s * n_n)
        else:
            nx_i_ls = []
            for j in range(n_n):
                nx_i_ls.append(nx_i[j].narrow(0, 0, self.ns_ls_ls[i][j]))
            nx_i = torch.cat(nx_i_ls).unsqueeze(0)
        a_i = one_hot(p[i].argmax().unsqueeze(0), self.n_fc)
        return F.relu(self.fc_x_layers[i](torch.cat([x[i].unsqueeze(0), nx_i], dim=1))) + \
               F.relu(self.fc_m_layers[i](m_i)) + a_i

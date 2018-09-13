import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import wargs
from gru import GRU
from tools.utils import *
from models.losser import *

import pdb

class NMT(nn.Module):

    def __init__(self, src_vocab_size, trg_vocab_size):

        super(NMT, self).__init__()

        self.encoder = Encoder(src_vocab_size, wargs.src_wemb_size, wargs.enc_hid_size)
        self.s_init = nn.Linear(wargs.enc_hid_size, wargs.dec_hid_size)
        self.tanh = nn.Tanh()
        self.ha = nn.Linear(2 * wargs.enc_hid_size, wargs.align_size)
        self.decoder = Decoder(trg_vocab_size)
        self.right_decoder = Decoder(trg_vocab_size, classifier=self.decoder.classifier)

    def init_state(self, h0_left):

        return self.tanh(self.s_init(h0_left))

    def init(self, xs, xs_mask=None, test=True):

        if test is True and not isinstance(xs, Variable):  # for decoding
            if wargs.gpu_id and not xs.is_cuda: xs = xs.cuda()
            xs = Variable(xs, requires_grad=False, volatile=True)

        x_s, h0_left = self.encoder(xs, xs_mask)
        s0 = self.init_state(h0_left)
        uh = self.ha(x_s)

        return s0, x_s, uh

    def forward(self, srcs, trgs, srcs_m, trgs_m, isAtt=False, test=False,
                ss_eps=1., oracles=None):

        # (max_slen_batch, batch_size, enc_hid_size)
        s0, srcs, uh = self.init(srcs, srcs_m, False)
        # reverse tgts
        reversed_tgts, tgts_mask_without_eos = self.reverse_batch_padded_seq(trgs, trgs_m)
        left_result, left_dec_states = self.decoder(s0, srcs, reversed_tgts, uh, srcs_m, tgts_mask_without_eos)

        tgts_valid_length = tc.squeeze(tc.sum(tgts_mask_without_eos, dim=0), 0).data.cpu().numpy().astype(int) # B
        seq_len, batch_size = tgts_mask_without_eos.size()
        reversed_left_dec_states = tc.transpose(left_dec_states, 0, 1) # B,S,H
        temp = []
        for s in xrange(batch_size):
            idx = Variable(tc.cat((tc.arange(tgts_valid_length[s] - 2, -1, -1).long(), tc.arange(tgts_valid_length[s] - 1, seq_len, 1).long())).cuda())
            temp.append(reversed_left_dec_states[s].index_select(0, idx)) # S,H
        reversed_left_dec_states = tc.transpose(tc.stack(temp, 0), 0, 1) # S,B,H

        right_result, _ = self.right_decoder(s0, srcs, trgs, uh, srcs_m, tgts_mask_without_eos, assist_states=reversed_left_dec_states)
        return left_result, right_result

    def reverse_batch_padded_seq(self, tgt, tgt_mask):
        # S,B => B,S
        tgt_t = tc.transpose(tgt, 0, 1)
        tgt_t_np = tgt_t.data.cpu().numpy().copy()
        batch_size, seq_len = tgt_t_np.shape

        # S,B => B,S
        tgt_mask_t = tc.transpose(tgt_mask, 0, 1)
        tgt_mask_t_np = tgt_mask_t.data.cpu().numpy().copy()

        # <bos> a b c d e <eos> 0 0 => <bos> e d c b a <eos> 0 0
        def reverse_seq(seq, seq_mask):
            left = 1
            right = seq_len - 1
            while seq[right] == 0:
                right -= 1
            if seq[right] == 3:
                seq_mask[right] = 0
                right -= 1
            while left < right:
                tmp = seq[right]
                seq[right] = seq[left]
                seq[left] = tmp
                left += 1
                right -= 1

        for s in xrange(batch_size):
            reverse_seq(tgt_t_np[s], tgt_mask_t_np[s])
        return Variable(tc.from_numpy(tgt_t_np.T).cuda()), Variable(tc.from_numpy(tgt_mask_t_np.T).cuda())

    def get_trainable_parameters(self):
        return ((n, p) for (n, p) in self.named_parameters())

class Encoder(nn.Module):

    '''
        Bi-directional Gated Recurrent Unit network encoder
    '''

    def __init__(self,
                 src_vocab_size,
                 input_size,
                 output_size,
                 with_ln=False,
                 prefix='Encoder', **kwargs):

        super(Encoder, self).__init__()

        self.output_size = output_size
        f = lambda name: str_cat(prefix, name)  # return 'Encoder_' + parameters name

        self.src_lookup_table = nn.Embedding(src_vocab_size, wargs.src_wemb_size, padding_idx=PAD)

        self.forw_gru = GRU(input_size, output_size, with_ln=with_ln, prefix=f('Forw'))
        self.back_gru = GRU(input_size, output_size, with_ln=with_ln, prefix=f('Back'))

    def forward(self, xs, xs_mask=None, h0=None):

        max_L, b_size = xs.size(0), xs.size(1)
        xs_e = xs if xs.dim() == 3 else self.src_lookup_table(xs)

        right = []
        h = h0 if h0 else Variable(tc.zeros(b_size, self.output_size), requires_grad=False)
        if wargs.gpu_id: h = h.cuda()
        for k in range(max_L):
            # (batch_size, src_wemb_size)
            h = self.forw_gru(xs_e[k], xs_mask[k] if xs_mask is not None else None, h)
            right.append(h)

        left = []
        h = h0 if h0 else Variable(tc.zeros(b_size, self.output_size), requires_grad=False)
        if wargs.gpu_id: h = h.cuda()
        for k in reversed(range(max_L)):
            h = self.back_gru(xs_e[k], xs_mask[k] if xs_mask is not None else None, h)
            left.append(h)

        right = tc.stack(right, dim=0)
        left = tc.stack(left[::-1], dim=0)
        # (slen, batch_size, 2*output_size)
        r1, r2 = tc.cat([right, left], -1), left[0]
        del right, left, h

        return r1, r2

class Attention(nn.Module):

    def __init__(self, dec_hid_size, align_size):

        super(Attention, self).__init__()
        self.align_size = align_size
        self.sa = nn.Linear(dec_hid_size, self.align_size)
        self.tanh = nn.Tanh()
        self.a1 = nn.Linear(self.align_size, 1)

    def forward(self, s_tm1, xs_h, uh, xs_mask=None):

        d1, d2, d3 = uh.size()
        # (b, dec_hid_size) -> (b, aln) -> (1, b, aln) -> (slen, b, aln) -> (slen, b)
        e_ij = self.a1(self.tanh(self.sa(s_tm1)[None, :, :] + uh)).squeeze(2)
        # 1, b
        e_ij_mean = e_ij.mean(0, keepdim=True)
        e_ij_std = e_ij.std(0, keepdim=True)
        e_ij = ((e_ij - e_ij_mean) / e_ij_std).exp()
        if np.isnan(e_ij.data.cpu().numpy()).any():
            print '[ERROR 1] e_ij contains nan'
        if xs_mask is not None: e_ij = e_ij * xs_mask

        # probability in each column: (slen, b)
        e_ij = e_ij / e_ij.sum(0)[None, :]
        if np.isnan(e_ij.data.cpu().numpy()).any():
            print '[ERROR 2] e_ij contains nan'

        # weighted sum of the h_j: (b, enc_hid_size)
        attend = (e_ij[:, :, None] * xs_h).sum(0)

        return e_ij, attend

class Decoder(nn.Module):

    def __init__(self, trg_vocab_size, max_out=True, classifier=None, with_ln=False):

        super(Decoder, self).__init__()

        self.max_out = max_out
        self.attention = Attention(wargs.dec_hid_size, wargs.align_size)
        self.assist_attention = Attention(wargs.dec_hid_size, wargs.align_size)
        self.assist_attention_w = nn.Linear(wargs.dec_hid_size, wargs.align_size)
        self.trg_lookup_table = nn.Embedding(trg_vocab_size, wargs.trg_wemb_size, padding_idx=PAD)
        self.tanh = nn.Tanh()
        self.gru = GRU(wargs.trg_wemb_size, wargs.dec_hid_size, enc_hid_size=2*wargs.enc_hid_size, with_ln=with_ln)

        out_size = 2 * wargs.out_size if max_out else wargs.out_size
        self.ls = nn.Linear(wargs.dec_hid_size, out_size)
        self.ly = nn.Linear(wargs.trg_wemb_size, out_size)
        self.lc = nn.Linear(2 * wargs.enc_hid_size, out_size)
        self.lc_assist = nn.Linear(wargs.dec_hid_size, out_size)

        self.write_f = nn.Linear(wargs.dec_hid_size, wargs.dec_hid_size)
        self.write_u = nn.Linear(wargs.dec_hid_size, wargs.dec_hid_size)

        if classifier:
            self.classifier = classifier
        else:
            self.classifier = Classifier(wargs.out_size, trg_vocab_size, self.trg_lookup_table if wargs.copy_trg_emb is True else None)

    def step(self, s_tm1, xs_h, uh, y_tm1, btg_xs_h=None, btg_uh=None, btg_xs_mask=None, xs_mask=None, y_mask=None, attend_assist=None):
        if not isinstance(y_tm1, tc.autograd.variable.Variable):
            if isinstance(y_tm1, int): y_tm1 = tc.Tensor([y_tm1]).long()
            elif isinstance(y_tm1, list): y_tm1 = tc.Tensor(y_tm1).long()
            if wargs.gpu_id: y_tm1 = y_tm1.cuda()
            y_tm1 = Variable(y_tm1, requires_grad=False, volatile=True)
            y_tm1 = self.trg_lookup_table(y_tm1)

        # (slen, batch_size), (batch_size, enc_hid_size)
        alpha_ij, attend = self.attention(s_tm1, xs_h, uh, xs_mask)
        s_t = self.gru(y_tm1, y_mask, s_tm1, attend, attend_assist=attend_assist)

        return attend, s_t, y_tm1, alpha_ij

    def forward(self, s_tm1, xs_h, ys, uh, xs_mask=None, ys_mask=None, isAtt=False, ss_eps=1., oracles=None, assist_states=None):

        tlen_batch_s, tlen_batch_c = [], []
        y_Lm1, b_size = ys.size(0), ys.size(1)
        if isAtt is True: attends = []
        # (max_tlen_batch - 1, batch_size, trg_wemb_size)
        ys_e = ys if ys.dim() == 3 else self.trg_lookup_table(ys)
        sent_logit, states, y_tm1_model = [], [], ys_e[0]
        for k in range(y_Lm1):

            if wargs.ss_type is not None and ss_eps < 1. and (wargs.greed_sampling or wargs.bleu_sampling):
                if wargs.greed_sampling is True:
                    if oracles is not None:     # joint word and sentence level
                        _seed = tc.Tensor(b_size, 1).bernoulli_()
                        _seed = Variable(_seed, requires_grad=False)
                        if wargs.gpu_id: _seed = _seed.cuda()
                        y_tm1_oracle = y_tm1_model * _seed + oracles[k] * (1. - _seed)
                    else:
                        y_tm1_oracle = y_tm1_model  # word-level oracle (w/o w/ noise)
                else:
                    y_tm1_oracle = oracles[k]   # sentence-level oracle

                _g = ss_eps * tc.ones(b_size, 1)
                _g = tc.bernoulli(_g)   # pick gold with the probability of ss_eps
                if wargs.gpu_id: _g = _g.cuda()
                _g = Variable(_g, requires_grad=False)
                y_tm1 = ys_e[k] * _g + y_tm1_oracle * (1. - _g)
            else:
                y_tm1 = ys_e[k]

            if isinstance(assist_states, Variable):
                # read
                assist_alpha, assist_ctx = self.assist_attention(s_tm1, assist_states, self.assist_attention_w(assist_states), ys_mask)
                #assist_ctx = Variable(tc.ones(b_size, wargs.dec_hid_size).cuda())
                attend, s_tm1, _, _ = self.step(s_tm1, xs_h, uh, y_tm1,
                                                xs_mask if xs_mask is not None else None,
                                                ys_mask[k] if ys_mask is not None else None,
                                                attend_assist=assist_ctx)
                # write
                assist_states = self.write_assist_state(s_tm1, assist_states, assist_alpha)
                assist_states = assist_states * ys_mask[:, :, None]
            else:
                attend, s_tm1, _, _ = self.step(s_tm1, xs_h, uh, y_tm1,
                                                xs_mask if xs_mask is not None else None,
                                                ys_mask[k] if ys_mask is not None else None)
            states.append(s_tm1)
            logit = self.step_out(s_tm1, y_tm1, attend, assist_c=assist_ctx if isinstance(assist_states, Variable) else None)
            sent_logit.append(logit)

            if wargs.ss_type is not None and ss_eps < 1. and wargs.greed_sampling is True:
                #logit = self.map_vocab(logit)
                logit = self.classifier.get_a(logit, noise=wargs.greed_gumbel_noise)
                y_tm1_model = logit.max(-1)[1]
                y_tm1_model = self.trg_lookup_table(y_tm1_model)

            #tlen_batch_c.append(attend)
            #tlen_batch_s.append(s_tm1)

            if isAtt is True: attends.append(alpha_ij)

        #s = tc.stack(tlen_batch_s, dim=0)
        #c = tc.stack(tlen_batch_c, dim=0)
        #del tlen_batch_s, tlen_batch_c

        #logit = self.step_out(s, ys_e, c)
        #if ys_mask is not None: logit = logit * ys_mask[:, :, None]  # !!!!
        logit = tc.stack(sent_logit, dim=0)
        logit = logit * ys_mask[:, :, None]  # !!!!
        states = tc.stack(states, dim=0)
        states = states * ys_mask[:,:,None]

        #del s, c
        results = (logit, states, tc.stack(attends, 0)) if isAtt is True else (logit, states)

        return results

    def write_assist_state(self, s_tm1, assist_states, assist_alpha):
        '''

        :param s_tm1: B, H
        :param assist_states: B, H
        :param assist_alpha: S, B, 1
        :return:
        '''
        ft, ut = F.sigmoid(self.write_f(s_tm1)), F.sigmoid(self.write_u(s_tm1))
        alpha_ij = assist_alpha[:, :, None]

        return assist_states * (1. - alpha_ij * ft[None, :, :]) + alpha_ij * ut[None, :, :]

    def step_out(self, s, y, c, assist_c=None):

        # (max_tlen_batch - 1, batch_size, dec_hid_size)
        if isinstance(assist_c, Variable):
            logit = self.ls(s) + self.ly(y) + self.lc(c) + self.lc_assist(assist_c)
        else:
            logit = self.ls(s) + self.ly(y) + self.lc(c)
        # (max_tlen_batch - 1, batch_size, out_size)

        if logit.dim() == 2:    # for decoding
            logit = logit.view(logit.size(0), logit.size(1)/2, 2)
        elif logit.dim() == 3:
            logit = logit.view(logit.size(0), logit.size(1), logit.size(2)/2, 2)

        return logit.max(-1)[0] if self.max_out else self.tanh(logit)

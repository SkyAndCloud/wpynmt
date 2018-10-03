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

EOS = 3

class Attention(nn.Module):

    def __init__(self, dec_hid_size, align_size, scale=False):

        super(Attention, self).__init__()
        self.align_size = align_size
        self.sa = nn.Linear(dec_hid_size, self.align_size)
        self.tanh = nn.Tanh()
        self.a1 = nn.Linear(self.align_size, 1)
        self.scale = scale

    def forward(self, s_tm1, xs_h, uh, xs_mask=None):

        d1, d2, d3 = uh.size()
        # (b, dec_hid_size) -> (b, aln) -> (1, b, aln) -> (slen, b, aln) -> (slen, b)
        e_ij = self.a1(self.tanh(self.sa(s_tm1)[None, :, :] + uh)).squeeze(2)
        if self.scale:
            # 1, b
            e_ij_mean = e_ij.mean(0, keepdim=True)
            e_ij_std = e_ij.std(0, keepdim=True)
            e_ij = ((e_ij - e_ij_mean) / e_ij_std).exp()
        else:
            e_ij = e_ij.exp()
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

class BackwardDecoder(nn.Module):

    def __init__(self, trg_vocab_size, max_out=True, classifier=None, with_ln=False):

        super(BackwardDecoder, self).__init__()

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
        sent_logit, states, y_tm1 = [], [], ys_e[0]
        y_mask = Variable(tc.FloatTensor([1] * b_size).cuda())
        mask_next_time = [False] * b_size
        tlen_batch_m = []

        for k in range(wargs.max_seq_len + 1):
            if isinstance(assist_states, Variable):
                # read
                assist_alpha, assist_ctx = self.assist_attention(s_tm1, assist_states, self.assist_attention_w(assist_states), ys_mask)
                #assist_ctx = Variable(tc.ones(b_size, wargs.dec_hid_size).cuda())
                attend, s_tm1, _, _ = self.step(s_tm1, xs_h, uh, y_tm1,
                                                xs_mask if xs_mask is not None else None,
                                                ys_mask,
                                                attend_assist=assist_ctx)
                # write
                # assist_states = self.write_assist_state(s_tm1, assist_states, assist_alpha)
                #assist_states = assist_states * ys_mask[:, :, None]
            else:
                attend, s_tm1, _, _ = self.step(s_tm1, xs_h, uh, y_tm1,
                                                xs_mask if xs_mask is not None else None,
                                                ys_mask)
            states.append(s_tm1)
            tlen_batch_m.append(y_mask)
            logit = self.step_out(s_tm1, y_tm1, attend, assist_c=assist_ctx if isinstance(assist_states, Variable) else None)
            sent_logit.append(logit)
            prob = self.classifier.softmax(self.classifier.get_a(logit))
            next_ces = tc.max(prob, 1)[1]
            y_tm1 = self.trg_lookup_table(next_ces)

            tmp_y_mask = y_mask.data.tolist()

            for i in range(b_size):
                if (tmp_y_mask[i] > 0.5) and (mask_next_time[i] == False):
                    tmp_y_mask[i] = 1.0
                else:
                    tmp_y_mask[i] = 0.0
            for i in range(b_size):
                if (next_ces.data[i] == EOS):
                    mask_next_time[i] = True
                else:
                    mask_next_time[i] = False
            reach_end = True
            for i in range(b_size):
                if (tmp_y_mask[i] > 0.5):
                    reach_end = False
            y_mask = Variable(tc.FloatTensor(tmp_y_mask).cuda())

            if reach_end == True:
                break

            if wargs.ss_type is not None and ss_eps < 1. and wargs.greed_sampling is True:
                #logit = self.map_vocab(logit)
                logit = self.classifier.get_a(logit, noise=wargs.greed_gumbel_noise)
                y_tm1_model = logit.max(-1)[1]
                y_tm1_model = self.trg_lookup_table(y_tm1_model)

            #tlen_batch_c.append(attend)
            #tlen_batch_s.append(s_tm1)


        #s = tc.stack(tlen_batch_s, dim=0)
        #c = tc.stack(tlen_batch_c, dim=0)
        #del tlen_batch_s, tlen_batch_c

        #logit = self.step_out(s, ys_e, c)
        #if ys_mask is not None: logit = logit * ys_mask[:, :, None]  # !!!!
        m = tc.stack(tlen_batch_m, dim=0)
        logit = tc.stack(sent_logit, dim=0)
        logit = logit * m[:, :, None]  # !!!!
        states = tc.stack(states, dim=0)
        states = states * m[:,:,None]

        #del s, c
        results = (logit, states, tc.stack(attends, 0)) if isAtt is True else (logit, states, m)

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

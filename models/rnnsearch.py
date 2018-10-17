import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pdb

import wargs
from gru import GRU
from tools.utils import *
from models.losser import *
from models.rnnsearch_backward_decoder import BackwardDecoder

class NMT(nn.Module):

    def __init__(self, src_vocab_size, trg_vocab_size):

        super(NMT, self).__init__()

        self.encoder = Encoder(src_vocab_size, wargs.src_wemb_size, wargs.enc_hid_size)
        self.s_init = nn.Linear(wargs.enc_hid_size, wargs.dec_hid_size)
        self.s_init_right_last = nn.Linear(wargs.enc_hid_size, wargs.dec_hid_size)
        self.tanh = nn.Tanh()
        self.ha = nn.Linear(wargs.enc_hid_size*2, wargs.align_size)
        self.right_decoder = Decoder(trg_vocab_size)
        self.decoder = BackwardDecoder(trg_vocab_size)

    def get_trainable_parameters(self):
        return ((n, p) for (n, p) in self.named_parameters())

    def init_state(self, h0_left, hn_right):
        return self.tanh(self.s_init(h0_left)), self.tanh(self.s_init_right_last(hn_right))

    def init(self, xs, xs_mask=None, test=True):
        if test is True and not isinstance(xs, Variable):  # for decoding
            if wargs.gpu_id and not xs.is_cuda: xs = xs.cuda()
            xs = Variable(xs, requires_grad=False, volatile=True)
        x_s, h0_left, hn_right = self.encoder(xs, xs_mask)
        s0, s0_bd = self.init_state(h0_left, hn_right)
        uh = self.ha(x_s)

        return s0, s0_bd, x_s, uh

    def forward(self, srcs, trgs, srcs_m, trgs_m, isAtt=False, test=False,
                ss_eps=1., oracles=None):
        # (max_slen_batch, batch_size, enc_hid_size)
        s0, s0_bd, srcs, uh = self.init(srcs, srcs_m, test)
        # reverse tgts
        reversed_tgts, tgts_mask_without_eos = self.reverse_batch_padded_seq(trgs, trgs_m)
        left_result, left_dec_states, left_dec_states_m = self.decoder(s0_bd, srcs, reversed_tgts, uh, srcs_m, tgts_mask_without_eos)
        right_result = self.right_decoder(s0, srcs, trgs, uh, srcs_m, tgts_mask_without_eos,
                                             assist_states=left_dec_states, assist_states_m=left_dec_states_m)
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
        self.back_gru = GRU(output_size, output_size, with_ln=with_ln, prefix=f('Back'))

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
            h = self.back_gru(right[k], xs_mask[k] if xs_mask is not None else None, h)
            left.append(h)

        right = tc.stack(right, dim=0)
        left = tc.stack(left[::-1], dim=0)
        # (slen, batch_size, 2*output_size)
        r1, r2, r3 = tc.cat([right, left], -1), left[0], right[-1]
        del right, left, h

        return r1, r2, r3

class Attention(nn.Module):

    def __init__(self, dec_hid_size, align_size):

        super(Attention, self).__init__()
        self.align_size = align_size
        self.sa = nn.Linear(dec_hid_size, self.align_size)
        self.tanh = nn.Tanh()
        self.maskSoftmax = MaskSoftmax()
        self.a1 = nn.Linear(self.align_size, 1)

    def forward(self, s_tm1, xs_h, uh, xs_mask=None):
        _check_tanh_sa = self.tanh(self.sa(s_tm1)[None, :, :] + uh)
        _check_a1_weight = self.a1.weight
        _check_a1 = self.a1(_check_tanh_sa).squeeze(2)

        e_ij = self.maskSoftmax(_check_a1, mask=xs_mask, dim=0)
        # weighted sum of the h_j: (b, enc_hid_size)
        attend = (e_ij[:, :, None] * xs_h).sum(0)

        return e_ij, attend

class Decoder(nn.Module):

    def __init__(self, trg_vocab_size, max_out=True):
        super(Decoder, self).__init__()
        self.max_out = max_out
        self.attention = Attention(wargs.dec_hid_size, wargs.align_size)
        self.assist_attention = Attention(wargs.dec_hid_size, wargs.align_size)
        self.assist_attention_w = nn.Linear(wargs.dec_hid_size, wargs.align_size)
        self.trg_lookup_table = nn.Embedding(trg_vocab_size, wargs.trg_wemb_size, padding_idx=PAD)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.gru1 = GRU(wargs.trg_wemb_size, wargs.dec_hid_size)
        self.gru2 = GRU(wargs.dec_hid_size * 3, wargs.dec_hid_size)

        out_size = 2 * wargs.out_size if max_out else wargs.out_size
        self.ls = nn.Linear(wargs.dec_hid_size, out_size)
        self.ly = nn.Linear(wargs.trg_wemb_size, out_size)
        self.lc = nn.Linear(wargs.enc_hid_size * 2 + wargs.dec_hid_size, out_size)
        self.classifier = Classifier(wargs.out_size, trg_vocab_size,
                                     self.trg_lookup_table if wargs.copy_trg_emb is True else None)

    def step(self, s_tm1, xs_h, uh, y_tm1, xs_mask=None, y_mask=None, assist_states=None, assist_states_m=None):
        if not isinstance(y_tm1, Variable):
            if isinstance(y_tm1, int): y_tm1 = tc.Tensor([y_tm1]).long()
            elif isinstance(y_tm1, list): y_tm1 = tc.Tensor(y_tm1).long()
            if wargs.gpu_id: y_tm1 = y_tm1.cuda()
            y_tm1 = Variable(y_tm1, requires_grad=False, volatile=True)
            y_tm1 = self.trg_lookup_table(y_tm1)

        if xs_mask is not None and not isinstance(xs_mask, Variable):
            xs_mask = Variable(xs_mask, requires_grad=False, volatile=True)
            if wargs.gpu_id: xs_mask = xs_mask.cuda()

        s_above = self.gru1(y_tm1, y_mask, s_tm1)
        alpha_ij, attend = self.attention(s_above, xs_h, uh, xs_mask)
        if isinstance(assist_states, Variable):
            assist_alpha, assist_context = self.assist_attention(s_above, assist_states, self.assist_attention_w(assist_states), assist_states_m)
        else:
            assist_alpha, assist_context = tc.zeros(wargs.max_seq_len, wargs.dec_hid_size), tc.zeros_like(s_tm1)
        context = tc.cat([attend, assist_context], -1)
        s_t = self.gru2(context, y_mask, s_above)

        return context, s_t, y_tm1, assist_alpha

    def forward(self, s_tm1, xs_h, ys, uh, xs_mask, ys_mask, assist_states=None, assist_states_m=None):
        y_Lm1, b_size = ys.size(0), ys.size(1)
        assert (xs_mask is not None) and (ys_mask is not None)
        # (max_tlen_batch - 1, batch_size, trg_wemb_size)
        ys_e = ys if ys.dim() == 3 else self.trg_lookup_table(ys)

        sent_logit, y_tm1_model = [], ys_e[0]
        for k in range(y_Lm1):
            y_tm1 = ys_e[k]
            context, s_tm1, y_tm1, assist_alpha = self.step(s_tm1, xs_h, uh, y_tm1, xs_mask, ys_mask[k], assist_states=assist_states, assist_states_m=assist_states_m)
            # TODO write mechanism
            logit = self.step_out(s_tm1, y_tm1, context)
            sent_logit.append(logit)
        logit = tc.stack(sent_logit, dim=0)
        logit = logit * ys_mask[:, :, None]  # !!!!

        return logit

    def step_out(self, s, y, c):
        logit = self.ls(s) + self.ly(y) + self.lc(c)

        if logit.dim() == 2:    # for decoding
            logit = logit.view(logit.size(0), logit.size(1)/2, 2)
        elif logit.dim() == 3:
            logit = logit.view(logit.size(0), logit.size(1), logit.size(2)/2, 2)

        return logit.max(-1)[0] if self.max_out else self.tanh(logit)

#( -*- coding: utf-8 -*-
from __future__ import division

import os
import re
import sys
import time
import numpy
import collections
from shutil import copyfile
from torch.autograd import Variable

import wargs
if wargs.model == 0 or wargs.model == 1: from searchs.nbs import *
elif wargs.model == 2: from searchs.nbs_ia import *
elif wargs.model == 3: from searchs.nbs_layers import *
elif wargs.model == 5: from searchs.nbs_sru import *
elif wargs.model == 6: from searchs.nbs_cyk import *
elif wargs.model == 8: from searchs.nbs_t2t import *

if wargs.search_mode == 1: from searchs.nbs import *
else: from searchs.cp import *

from tools.utils import *
from tools.bleu import bleu_file
from tools.multibleu import print_multi_bleu
import uniout

numpy.set_printoptions(threshold=numpy.nan)

class Translator(object):

    def __init__(self, model, svcb_i2w=None, tvcb_i2w=None, search_mode=None, thresh=None, lm=None,
                 ngram=None, ptv=None, k=None, noise=None, print_att=False):

        self.svcb_i2w = svcb_i2w
        self.tvcb_i2w = tvcb_i2w
        self.lm = lm
        self.ptv = ptv
        self.noise = noise
        self.model = model
        self.ngram = ngram
        self.thresh = thresh
        self.print_att = print_att
        self.k = k if k else wargs.beam_size
        self.search_mode = search_mode if search_mode else wargs.search_mode

        if self.search_mode == 0:
            self.greedy = Greedy(self.tvcb_i2w)
        elif self.search_mode == 1:
            self.nbs = Nbs(model, self.tvcb_i2w, k=self.k, noise=self.noise, print_att=print_att)
        elif self.search_mode == 2:
            self.wcp = Wcp(model, self.tvcb_i2w, k=self.k, print_att=print_att)

    def trans_onesent(self, s):

        trans_start = time.time()

        if self.search_mode == 0:
            trans = self.greedy.greedy_trans(s)
        elif self.search_mode == 1:
            batch_tran_cands = self.nbs.beam_search_trans(s)
        #elif self.search_mode == 2: (trans, ids), loss = self.wcp.cube_prune_trans(s)
        elif self.search_mode == 2:
            batch_tran_cands = self.wcp.cube_prune_trans(s)
        trans, loss, attent_matrix = batch_tran_cands[0][0] # first sent, best cand
        trans = trans[::-1]
        trans, ids = filter_reidx(trans, self.tvcb_i2w)

        #spend = time.time() - trans_start
        #wlog('Word-Level spend: {} / {} = {}'.format(
        #    format_time(spend), len(ids), format_time(spend / len(ids))))

        # attent_matrix: (trgL, srcL) numpy
        return trans, ids, attent_matrix

    def trans_samples(self, srcs, trgs, spos=None):

        if isinstance(srcs, tc.autograd.variable.Variable): t_srcs = srcs.data
        if isinstance(trgs, tc.autograd.variable.Variable): t_trgs = trgs.data

        # srcs: (sample_size, max_sLen)
        for idx in range(len(t_srcs)):

            s_filter = sent_filter(list(t_srcs[idx]))
            src_sent = idx2sent(s_filter, self.svcb_i2w)
            if len(src_sent) == 2: src_sent, ori_src_toks = src_sent
            wlog('\n[{:3}] {}'.format('Src', src_sent))
            t_filter = sent_filter(list(t_trgs[idx]))
            ref_sent = idx2sent(t_filter, self.tvcb_i2w)
            if len(ref_sent) == 2: ref_sent, ori_ref_toks = ref_sent
            wlog('[{:3}] {}'.format('Ref', ref_sent))

            if wargs.model == 8:
                src_seq, src_pos = srcs[idx], spos[idx]
                non_zero_idx = src_seq.data.cpu().nonzero().squeeze().numpy()
                src_seq, src_pos = src_seq[non_zero_idx], src_pos[non_zero_idx]
                s_filter = (src_seq[None, :], src_pos[None, :])

            trans, ids, attent_matrix = self.trans_onesent(s_filter)

            if len(trans) == 2:
                trans, trg_toks = trans
                trans_subwords = '###'.join(trg_toks)
                wlog('[{:3}] {}'.format('Sub', trans_subwords))
            else: src_toks, trg_toks = src_sent.split(' '), trans.split(' ')

            if wargs.with_bpe is True:
                wlog('[{:3}] {}'.format('Bpe', trans))
                #trans = trans.replace('@@ ', '')
                trans = re.sub('(@@ )|(@@ ?$)', '', trans)

            if self.print_att is True:
                if isinstance(self.svcb_i2w, dict):
                    src_toks = [self.svcb_i2w[wid] for wid in s_filter]
                else:
                    src = self.svcb_i2w.decode(s_filter)
                    if len(src) == 2: src_sent, src_toks = src
                print_attention_text(attent_matrix, src_toks, trg_toks, isP=True)
                plot_attention(attent_matrix, src_toks, trg_toks, 'att.svg')

            #trans = re.sub('( ##AT##)|(##AT## )', '', trans)
            wlog('[{:3}] {}'.format('Out', trans))

    def force_decoding(self, batch_tst_data):

        batch_count = len(batch_tst_data)
        point_every, number_every = int(math.ceil(batch_count/100)), int(math.ceil(batch_count/10))
        attent_matrixs, trg_toks = [], []
        for batch_idx in range(batch_count):
            _, srcs, ttrgs_for_files, _, srcs_m, trg_mask_for_files = batch_tst_data[batch_idx]
            trgs, trgs_m = ttrgs_for_files[0], trg_mask_for_files[0]
            # src: ['我', '爱', '北京', '天安门']
            # trg: ['<b>', 'i', 'love', 'beijing', 'tiananmen', 'square', '<e>']
            # feed ['<b>', 'i', 'love', 'beijing', 'tiananmen', 'square']
            # must feed first <b>, because feed previous word to get alignment of next word 'i' !!!
            outputs = self.model(srcs, trgs[:-1], srcs_m, trgs_m[:-1], isAtt=True, test=True)
            if len(outputs) == 2: (outputs, _checks) = outputs
            if len(outputs) == 2: (outputs, attends) = outputs
            # attends: (trg_maxL-1, src_maxL, B)

            attent_matrixs.extend(list(attends.permute(2, 0, 1).cpu().data.numpy()))
            # (trg_maxL-2, B) -> (B, trg_maxL-2)
            trg_toks.extend(list(trgs[1:-1].permute(1, 0).cpu().data.numpy()))

            if numpy.mod(batch_idx + 1, point_every) == 0: wlog('.', False)
            if numpy.mod(batch_idx + 1, number_every) == 0: wlog('{}'.format(batch_idx + 1), False)
        wlog('')

        assert len(attent_matrixs) == len(trg_toks)
        return attent_matrixs, trg_toks

    def single_trans_file(self, src_input_data, src_labels_fname=None, batch_tst_data=None):

        batch_count = len(src_input_data)   # number of batchs, here is sentences number
        point_every, number_every = int(math.ceil(batch_count/100)), int(math.ceil(batch_count/10))
        total_trans = []
        total_aligns = [] if self.print_att is True else None
        sent_no, words_cnt = 0, 0
        if wargs.word_piece is True: total_trans_subwords = []

        fd_attent_matrixs, trgs = None, None
        if batch_tst_data is not None:
            wlog('\nStarting force decoding ...')
            fd_attent_matrixs, trgs = self.force_decoding(batch_tst_data)
            wlog('Finish force decoding ...')

        trans_start = time.time()
        for bid in range(batch_count):
            n_bid_src = src_input_data[bid]
            # n_bid_src: (idxs, tsrcs, tspos, lengths, src_mask)
            # idxs, tsrcs, tspos, ttrgs_for_files, ttpos_for_files, lengths, src_mask, trg_mask_for_files
            batch_srcs_LB = n_bid_src[1]
            #batch_srcs_LB = src_input_data[bid][1] # (dxs, tsrcs, lengths, src_mask)
            #batch_srcs_LB = batch_srcs_LB.squeeze()
            for no in range(batch_srcs_LB.size(1)): # batch size, 1 for valid
                s_filter = sent_filter(list(batch_srcs_LB[:,no].data))

                if wargs.word_piece is True: trans_subwords = []
                if src_labels_fname is not None:
                    assert self.print_att is None, 'split sentence does not suport print attention'
                    # split by segment labels file
                    segs = self.segment_src(s_filter, labels[bid].strip().split(' '))
                    trans = []
                    for seg in segs:
                        seg_trans, ids, _ = self.trans_onesent(seg)
                        if len(seg_trans) == 2:
                            seg_trans, seg_subwords = seg_trans
                            trans_subwords.append('###'.join(seg_subwords))
                        else: _ = seg_trans.split(' ')
                        words_cnt += len(ids)
                        trans.append(seg_trans)
                    # merge by order
                    trans = ' '.join(trans)
                    if wargs.word_piece is True: trans_subwords = '###'.join(trans_subwords)
                else:
                    if fd_attent_matrixs is None:   # need translate
                        trans, ids, attent_matrix = self.trans_onesent(n_bid_src)
                        if len(trans) == 2:
                            trans, trg_toks = trans
                            trans_subwords = '###'.join(trg_toks)
                        else: trg_toks = trans.split(' ')
                        if trans == '': wlog('What ? null translation ... !')
                        words_cnt += len(ids)
                    else:
                        # attention: feed previous word -> get the alignment of next word !!!
                        attent_matrix = fd_attent_matrixs[bid] # do not remove <b>
                        #print attent_matrix
                        trg_toks = sent_filter(trgs[bid]) # remove <b> and <e>
                        trg_toks = [self.tvcb_i2w[wid] for wid in trg_toks]
                        trans = ' '.join(trg_toks)
                        words_cnt += len(trg_toks)

                    # get alignment from attent_matrix for one translation
                    if attent_matrix is not None:
                        # maybe generate null translation, fault-tolerant here
                        if isinstance(attent_matrix, list) and len(attent_matrix) == 0: alnStr = ''
                        else:
                            if isinstance(self.svcb_i2w, dict):
                                src_toks = [self.svcb_i2w[wid] for wid in s_filter]
                            else:
                                #print type(self.svcb_i2w)
                                # <class 'tools.text_encoder.SubwordTextEncoder'>
                                src_toks = self.svcb_i2w.decode(s_filter)
                                if len(src_toks) == 2: _, src_toks = src_toks
                                #print src_toks
                            # attent_matrix: (trgL, srcL) numpy
                            alnStr = print_attention_text(attent_matrix, src_toks, trg_toks)
                        total_aligns.append(alnStr)

                total_trans.append(trans)
                if wargs.word_piece is True: total_trans_subwords.append(trans_subwords)
                if numpy.mod(sent_no + 1, point_every) == 0: wlog('.', False)
                if numpy.mod(sent_no + 1, number_every) == 0: wlog('{}'.format(sent_no + 1), False)

                sent_no += 1
        wlog('')

        if self.search_mode == 1:
            C = self.nbs.C
            if C[0] != 0:
                wlog('Average location of bp [{}/{}={:6.4f}]'.format(C[1], C[0], C[1] / C[0]))
                wlog('Step[{}] stepout[{}]'.format(*C[2:]))

        if self.search_mode == 2:
            C = self.wcp.C
            if C[0] != 0 and C[2] != 0:
                wlog('Average Merging Rate [{}/{}={:6.4f}]'.format(C[1], C[0], C[1] / C[0]))
                wlog('Average location of bp [{}/{}={:6.4f}]'.format(C[3], C[2], C[3] / C[2]))
                wlog('Step[{}] stepout[{}]'.format(*C[4:]))

        spend = time.time() - trans_start
        if words_cnt == 0: wlog('What ? No words generated when translating one file !!!')
        else:
            wlog('Word-Level spend: [{}/{} = {}/w], [{}/{:7.2f}s = {:7.2f} w/s]'.format(
                format_time(spend), words_cnt, format_time(spend / words_cnt),
                words_cnt, spend, words_cnt/spend))

        wlog('Done ...')
        if total_aligns is not None: total_aligns = '\n'.join(total_aligns) + '\n'
        total_trans_subwords = '\n'.join(total_trans_subwords) if wargs.word_piece else None
        return '\n'.join(total_trans) + '\n', total_aligns, total_trans_subwords

    def segment_src(self, src_list, labels_list):

        #print len(src_list), len(labels_list)
        assert len(src_list) == len(labels_list)
        segments, seg = [], []
        for i in range(len(src_list)):
            c, l = src_list[i], labels_list[i]
            if l == 'S':
                segments.append([c])
            elif l == 'E':
                seg.append(c)
                segments.append(seg)
                seg = []
            elif l == 'B':
                if len(seg) > 0: segments.append(seg)
                seg = []
                seg.append(c)
            else:
                seg.append(c)

        return segments

    def write_file_eval(self, out_fname, trans, data_prefix, alns=None, subw=None):

        if alns is not None:
            fout_aln = open('{}.aln'.format(out_fname), 'w')    # valids/trans
            fout_aln.writelines(alns)
            fout_aln.close()

        if subw is not None:
            fout_subw = open('{}.subword'.format(out_fname), 'w')    # valids/trans
            fout_subw.writelines(subw)
            fout_subw.close()

        fout = open(out_fname, 'w')    # valids/trans
        fout.writelines(trans)
        fout.close()

        ref_fpaths = []
        # *.ref
        ref_fpath = '{}{}.{}'.format(wargs.val_tst_dir, data_prefix, wargs.val_ref_suffix)
        if os.path.exists(ref_fpath): ref_fpaths.append(ref_fpath)
        for idx in range(wargs.ref_cnt):
            # *.ref0, *.ref1, ...
            ref_fpath = '{}{}.{}{}'.format(wargs.val_tst_dir, data_prefix, wargs.val_ref_suffix, idx)
            if not os.path.exists(ref_fpath): continue
            ref_fpaths.append(ref_fpath)

        assert os.path.exists(out_fname), 'translation do not exist ...'
        if wargs.with_bpe is True:
            bpe_fname = '{}.bpe'.format(out_fname)
            wlog('copy {} to {} ... '.format(out_fname, bpe_fname), 0)
            #os.system('cp {} {}.bpe'.format(out_fname, out_fname))
            copyfile(out_fname, bpe_fname)
            assert os.path.exists(bpe_fname), 'bpe file do not exist ...'
            wlog('done')
            wlog("sed -r 's/(@@ )|(@@ ?$)//g' {} > {} ... ".format(bpe_fname, out_fname), 0)
            #os.system("sed -r 's/(@@ )|(@@ ?$)//g' {} > {}".format(bpe_fname, out_fname))
            proc_bpe(bpe_fname, out_fname)
            wlog('done')

        # Luong: remove "rich-text format" --> rich ##AT##-##AT## text format
        #os.system("sed -r -i 's/( ##AT##)|(##AT## )//g' {}".format(out_fname))
        #wlog("sed -r -i 's/( ##AT##)|(##AT## )//g' {}".format(out_fname))
        if wargs.with_postproc is True:
            opost_name = '{}.opost'.format(out_fname)
            wlog('copy {} to {} ... '.format(out_fname, opost_name), 0)
            #os.system('cp {} {}'.format(out_fname, opost_name))
            copyfile(out_fname, opost_name)
            assert os.path.exists(opost_name), 'opost file do not exist ...'
            wlog('done')
            wlog("sh postproc.sh {} {}".format(opost_name, out_fname))
            os.system("sh postproc.sh {} {}".format(opost_name, out_fname))
            wlog('done')
            mteval_bleu_opost = bleu_file(opost_name, ref_fpaths, cased=wargs.cased)
            os.rename(opost_name, "{}_{}.txt".format(opost_name, mteval_bleu_opost))

        mteval_bleu = bleu_file(out_fname, ref_fpaths, cased=wargs.cased)

        multi_bleu = print_multi_bleu(out_fname, ref_fpaths, cased=wargs.cased)
        #mteval_bleu = bleu_file(out_fname + '.seg.plain', ref_fpaths)
        if wargs.char is True:
            c_mteval_bleu = bleu_file(out_fname, ref_fpaths, cased=wargs.cased, char=True)
            c_multi_bleu = print_multi_bleu(out_fname, ref_fpaths, cased=wargs.cased, char=True)
            os.rename(out_fname, "{}_{}_{}_c_{}_{}.txt".format(
                out_fname, mteval_bleu, multi_bleu, c_mteval_bleu, c_multi_bleu))
        else: os.rename(out_fname, "{}_{}_{}.txt".format(out_fname, mteval_bleu, multi_bleu))

        if wargs.use_multi_bleu is False:
            if wargs.char is False: final_bleu = mteval_bleu
            else: final_bleu = c_mteval_bleu
        else:
            if wargs.char is False: final_bleu = multi_bleu
            else: final_bleu = c_multi_bleu

        return final_bleu
        #return mteval_bleu_opost if wargs.with_postproc is True else mteval_bleu

    def trans_tests(self, tests_data, eid, bid):

        for _, test_prefix in zip(tests_data, wargs.tests_prefix):

            wlog('\nTranslating test dataset {}'.format(test_prefix))
            label_fname = '{}{}/{}.label'.format(wargs.val_tst_dir, wargs.seg_val_tst_dir,
                                                 test_prefix) if wargs.segments else None
            trans, alns, subw = self.single_trans_file(tests_data[test_prefix], label_fname)

            outprefix = wargs.dir_tests + '/' + test_prefix + '/trans'
            test_out = "{}_e{}_upd{}_b{}m{}_bch{}".format(
                outprefix, eid, bid, self.k, self.search_mode, wargs.with_batch)

            _ = self.write_file_eval(test_out, trans, test_prefix, alns, subw)

    def trans_eval(self, valid_data, eid, bid, model_file, tests_data):

        wlog('\nTranslating validation dataset {}{}.{}'.format(wargs.val_tst_dir, wargs.val_prefix, wargs.val_src_suffix))
        label_fname = '{}{}/{}.label'.format(wargs.val_tst_dir, wargs.seg_val_tst_dir,
                                             wargs.val_prefix) if wargs.segments else None
        trans, alns, subw = self.single_trans_file(valid_data, label_fname)

        outprefix = wargs.dir_valid + '/trans'
        valid_out = "{}_e{}_upd{}_b{}m{}_bch{}".format(
            outprefix, eid, bid, self.k, self.search_mode, wargs.with_batch)

        bleu_score = self.write_file_eval(valid_out, trans, wargs.val_prefix, alns, subw)

        bleu_scores_fname = '{}/train_bleu.log'.format(wargs.dir_valid)
        bleu_scores = [0.]
        if os.path.exists(bleu_scores_fname):
            with open(bleu_scores_fname) as f:
                for line in f:
                    s_bleu = line.split(':')[-1].strip()
                    bleu_scores.append(float(s_bleu))

        wlog('\nCurrent [{}] - Best History [{}]'.format(bleu_score, max(bleu_scores)))
        if bleu_score > max(bleu_scores):   # better than history
            copyfile(model_file, wargs.best_model)
            wlog('Better, cp {} {}'.format(model_file, wargs.best_model))
            bleu_content = 'epoch [{}], batch[{}], BLEU score*: {}'.format(eid, bid, bleu_score)
            if wargs.final_test is False and tests_data is not None: self.trans_tests(tests_data, eid, bid)
        else:
            wlog('Worse')
            bleu_content = 'epoch [{}], batch[{}], BLEU score : {}'.format(eid, bid, bleu_score)

        append_file(bleu_scores_fname, bleu_content)

        sfig = '{}.{}'.format(outprefix, 'sfig')
        sfig_content = ('{} {} {} {} {}').format(eid, bid, self.search_mode, self.k, bleu_score)
        append_file(sfig, sfig_content)

        if wargs.save_one_model and os.path.exists(model_file) is True:
            os.remove(model_file)
            wlog('Saving one model, so delete {}\n'.format(model_file))

        return bleu_score

if __name__ == "__main__":
    import sys
    res = valid_bleu(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    wlog(res)

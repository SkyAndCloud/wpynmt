import torch as tc
from torch import cuda

import wargs
from tools.inputs import Input
from tools.utils import init_dir, wlog, _load_model
from tools.optimizer import Optim
from inputs_handler import *
import tools.crash_on_ipy

# Check if CUDA is available
if cuda.is_available():
    wlog('CUDA is available, specify device by gpu_id argument (i.e. gpu_id=[3])')
else:
    wlog('Warning: CUDA is not available, train on CPU')

if wargs.gpu_id:
    cuda.set_device(wargs.gpu_id[0])
    wlog('Using GPU {}'.format(wargs.gpu_id[0]))

if wargs.model == 0: from models.groundhog import *
elif wargs.model == 1: from models.rnnsearch import *
elif wargs.model == 2: from models.rnnsearch_ia import *
elif wargs.model == 3: from models.ran_agru import *
elif wargs.model == 4: from models.rnnsearch_rn import *
elif wargs.model == 5: from models.nmt_sru import *
elif wargs.model == 6: from models.nmt_cyk import *
elif wargs.model == 7: from models.non_local import *

from trainer import *
from translate import Translator

import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.enabled = True

tc.manual_seed(1111)

def main():

    #if wargs.ss_type is not None: assert wargs.model == 1, 'Only rnnsearch support schedule sample'
    init_dir(wargs.dir_model)
    init_dir(wargs.dir_valid)

    src = os.path.join(wargs.dir_data, '{}.{}'.format(wargs.train_prefix, wargs.train_src_suffix))
    trg = os.path.join(wargs.dir_data, '{}.{}'.format(wargs.train_prefix, wargs.train_trg_suffix))
    vocabs = {}
    if wargs.word_piece is True:
        wlog('\n[w/Subword] Preparing source vocabulary from {} ... '.format(src))
        src_vocab = get_or_generate_vocab(src, wargs.src_dict, max_seq_len=wargs.max_seq_len)
        wlog('\n[w/Subword] Preparing target vocabulary from {} ... '.format(trg))
        trg_vocab = get_or_generate_vocab(trg, wargs.trg_dict, max_seq_len=wargs.max_seq_len)
    else:
        wlog('\n[o/Subword] Preparing source vocabulary from {} ... '.format(src))
        src_vocab = extract_vocab(src, wargs.src_dict, wargs.src_dict_size, wargs.max_seq_len)
        wlog('\n[o/Subword] Preparing target vocabulary from {} ... '.format(trg))
        trg_vocab = extract_vocab(trg, wargs.trg_dict, wargs.trg_dict_size, wargs.max_seq_len)
    src_vocab_size, trg_vocab_size = src_vocab.size(), trg_vocab.size()
    wlog('Vocabulary size: |source|={}, |target|={}'.format(src_vocab_size, trg_vocab_size))
    vocabs['src'], vocabs['trg'] = src_vocab, trg_vocab

    wlog('\nPreparing training set from {} and {} ... '.format(src, trg))
    trains = {}
    train_src_tlst, train_trg_tlst = wrap_data(wargs.dir_data, wargs.train_prefix,
                                               wargs.train_src_suffix, wargs.train_trg_suffix,
                                               src_vocab, trg_vocab, max_seq_len=wargs.max_seq_len)
    '''
    list [torch.LongTensor (sentence), torch.LongTensor, torch.LongTensor, ...]
    no padding
    '''
    batch_train = Input(train_src_tlst, train_trg_tlst, wargs.batch_size, batch_sort=True)
    wlog('Sentence-pairs count in training data: {}'.format(len(train_src_tlst)))

    batch_valid = None
    if wargs.val_prefix is not None:
        val_src_file = '{}{}.{}'.format(wargs.val_tst_dir, wargs.val_prefix, wargs.val_src_suffix)
        val_trg_file = '{}{}.{}'.format(wargs.val_tst_dir, wargs.val_prefix, wargs.val_ref_suffix)
        wlog('\nPreparing validation set from {} and {} ... '.format(val_src_file, val_trg_file))
        valid_src_tlst, valid_trg_tlst = wrap_data(wargs.val_tst_dir, wargs.val_prefix,
                                                   wargs.val_src_suffix, wargs.val_ref_suffix,
                                                   src_vocab, trg_vocab,
                                                   shuffle=False, sort_data=False,
                                                   max_seq_len=wargs.dev_max_seq_len)
        batch_valid = Input(valid_src_tlst, valid_trg_tlst, 1, volatile=True, batch_sort=False)

    batch_tests = None
    if wargs.tests_prefix is not None:
        assert isinstance(wargs.tests_prefix, list), 'Test files should be list.'
        init_dir(wargs.dir_tests)
        batch_tests = {}
        for prefix in wargs.tests_prefix:
            init_dir(wargs.dir_tests + '/' + prefix)
            test_file = '{}{}.{}'.format(wargs.val_tst_dir, prefix, wargs.val_src_suffix)
            wlog('\nPreparing test set from {} ... '.format(test_file))
            test_src_tlst, _ = wrap_tst_data(test_file, src_vocab)
            batch_tests[prefix] = Input(test_src_tlst, None, 1, volatile=True, batch_sort=False)
    wlog('\n## Finish to Prepare Dataset ! ##\n')

    if wargs.model == 8:
        from models.transformer import Transformer
        #from transformer.Models import Transformer
        nmtModel = Transformer(
            src_vocab_size,
            trg_vocab_size,
            wargs.max_seq_len,
            proj_share_weight=wargs.proj_share_weight,
            embs_share_weight=wargs.embs_share_weight,
            d_model=wargs.d_model,
            d_word_vec=wargs.d_word_vec,
            d_inner_hid=wargs.d_inner_hid,
            n_layers=wargs.n_layers,
            n_head=wargs.n_head,
            dropout=wargs.drop_rate)

        def get_criterion(vocab_size):
            weight = tc.ones(vocab_size)
            weight[PAD] = 0
            return nn.CrossEntropyLoss(weight, size_average=False)
        crit = get_criterion(trg_vocab_size)
    else: nmtModel = NMT(src_vocab_size, trg_vocab_size)

    if wargs.pre_train is not None:

        assert os.path.exists(wargs.pre_train)

        _dict = _load_model(wargs.pre_train)
        # initializing parameters of interactive attention model
        class_dict = None
        if len(_dict) == 4: model_dict, eid, bid, optim = _dict
        elif len(_dict) == 5:
            model_dict, class_dict, eid, bid, optim = _dict
        for name, param in nmtModel.named_parameters():
            if name in model_dict:
                param.requires_grad = not wargs.fix_pre_params
                param.data.copy_(model_dict[name])
                wlog('{:7} -> grad {}\t{}'.format('Model', param.requires_grad, name))
            elif name.endswith('map_vocab.weight'):
                if class_dict is not None:
                    param.requires_grad = not wargs.fix_pre_params
                    param.data.copy_(class_dict['map_vocab.weight'])
                    wlog('{:7} -> grad {}\t{}'.format('Model', param.requires_grad, name))
            elif name.endswith('map_vocab.bias'):
                if class_dict is not None:
                    param.requires_grad = not wargs.fix_pre_params
                    param.data.copy_(class_dict['map_vocab.bias'])
                    wlog('{:7} -> grad {}\t{}'.format('Model', param.requires_grad, name))
            else: init_params(param, name, True)

        wargs.start_epoch = eid + 1

    else:
        if wargs.model != 8:
            for n, p in nmtModel.named_parameters(): init_params(p, n, True)
        optim = Optim(
            wargs.opt_mode, wargs.learning_rate, wargs.max_grad_norm,
            learning_rate_decay=wargs.learning_rate_decay,
            start_decay_from=wargs.start_decay_from,
            last_valid_bleu=wargs.last_valid_bleu, model=wargs.model
        )

    if wargs.gpu_id is not None:
        #nmtModel.cuda()
        #nmtModel = tc.nn.DataParallel(nmtModel, device_ids=wargs.gpu_id, dim=1)
        wlog('Push model onto GPU {} ... '.format(wargs.gpu_id), 0)
        nmtModel.cuda()
    else:
        wlog('Push model onto CPU ... ', 0)
        nmtModel.cpu()

    wlog('done.')

    wlog(nmtModel)
    wlog(optim)
    pcnt1 = len([p for p in nmtModel.parameters()])
    pcnt2 = sum([p.nelement() for p in nmtModel.parameters()])
    wlog('Parameters number: {}/{}'.format(pcnt1, pcnt2))

    wlog('\n' + '*' * 30 + ' Trainable parameters ' + '*' * 30)
    #  for n, p in nmtModel.get_trainable_parameters(): wlog(n)
    if wargs.model == 8: optim.init_optimizer((p for n, p in nmtModel.get_trainable_parameters()))
    else: optim.init_optimizer(nmtModel.parameters())

    trainer = Trainer(nmtModel, batch_train, vocabs, optim, batch_valid, batch_tests)

    trainer.train()


if __name__ == "__main__":

    main()

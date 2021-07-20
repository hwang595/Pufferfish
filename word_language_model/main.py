# coding: utf-8
import argparse
import time
import math
import random
import os
import logging

import numpy as np

import torch
import torch.nn as nn
import torch.onnx
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

import data
import model

# helper function because otherwise non-empty strings
# evaluate as True
def bool_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--evaluate', type=bool_string, default=False,
                    help='if or not to evaluate the save model and exit.')
parser.add_argument('--ckpt-path', type=str, default="/home/ubuntu/low-rank-ml/word_language_model/vanilla_model_seed1111_best.pt",
                    help='checkpoint path to evaluate the model')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--warmup', type=bool_string, default=True, 
                    help="wether or not to use full-rank warmup")
parser.add_argument('--warmup-epoch', type=int, default=5, 
                    help="wether or not to use full-rank warmup")
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument('--dry-run', action='store_true',
                    help='verify the code and the model')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
def seed(seed):
    # seed = 1234
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #TODO: Do we need deterministic in cudnn ? Double check
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Seeded everything")

#torch.manual_seed(args.seed)
seed(seed=args.seed)

if torch.cuda.is_available():
    if not args.cuda:
        logger.info("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def decompose_vanilla_model(vanilla_model, low_rank_model, rank_ratio=0.25):
    collected_weights = []
    for p_index, (name, param) in enumerate(vanilla_model.state_dict().items()):
        if "rnn." in name and len(param.size()) == 2:
            rank = min(param.size()[0], param.size()[1])
            sliced_rank = int(rank * rank_ratio)
            u, s, v = torch.svd(param)
            u_weight = u * torch.sqrt(s) # alternative implementation: u_weight_alt = torch.mm(u, torch.diag(torch.sqrt(s)))
            v_weight = torch.sqrt(s) * v # alternative implementation: v_weight_alt = torch.mm(torch.diag(torch.sqrt(s)), v.t())
            #print("layer indeix: {}, dist u u_alt:{}, dist v v_alt: {}".format(item_index, torch.dist(u_weight, u_weight_alt), torch.dist(v_weight.t(), v_weight_alt)))
            #print("layer indeix: {}, dist u u_alt:{}, dist v v_alt: {}".format(item_index, torch.equal(u_weight, u_weight_alt), torch.equal(v_weight.t(), v_weight_alt)))
            u_weight_sliced, v_weight_sliced = u_weight[:, 0:sliced_rank], v_weight[:, 0:sliced_rank]
            collected_weights.append(u_weight_sliced)
            collected_weights.append(v_weight_sliced.t())
        else:
            collected_weights.append(param)
            
    #for cw_index, cw in enumerate(collected_weights):
    #     print("cw_index: {}, cw: {}".format(cw_index, cw.size()))
         
    reconstructed_state_dict = {}
    model_counter = 0
    for p_index, (name, param) in enumerate(low_rank_model.state_dict().items()):
        assert param.size() == collected_weights[model_counter].size()
        reconstructed_state_dict[name] = collected_weights[model_counter]
        model_counter += 1
    low_rank_model.load_state_dict(reconstructed_state_dict)
    return low_rank_model


eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

def param_counter(model):
    num_params = 0
    for param_index, (param_name, param) in enumerate(model.named_parameters()):
        logger.info("param index : {}, param name: {}, param: {}".format(param_index, param_name, param.size()))
        num_params += param.nelement()
    return num_params

ntokens = len(corpus.dictionary)
if args.model == 'Transformer':
    model = model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
else:
    vanilla_model = model.RNNModel(args.model, ntokens, 
                                    args.emsize, args.nhid, 
                                    args.nlayers, args.dropout, 
                                    args.tied).to(device)
    low_rank_model = model.LowRankRNNModel(args.model, ntokens, 
                                args.emsize, args.nhid, 
                                args.nlayers, args.dropout, 
                                rank_ratio=0.25,
                                tie_weights=args.tied,
                                ).to(device)
    logger.info("Num params for vanilla model: {}, Num params for low-rank model: {}".format(param_counter(model=vanilla_model),
                                param_counter(model=low_rank_model)))

criterion = nn.NLLLoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    low_rank_model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = low_rank_model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            if args.model == 'Transformer':
                output = low_rank_model(data)
                output = output.view(-1, ntokens)
            else:
                output, hidden = low_rank_model(data, hidden)
                hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)
    

def evaluate_vanilla(data_source):
    # Turn on evaluation mode which disables dropout.
    vanilla_model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = vanilla_model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            if args.model == 'Transformer':
                output = vanilla_model(data)
                output = output.view(-1, ntokens)
            else:
                output, hidden = vanilla_model(data, hidden)
                hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)


def train():
    # Turn on training mode which enables dropout.
    low_rank_model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = low_rank_model.init_hidden(args.batch_size)

    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        low_rank_model.zero_grad()

        if args.model == 'Transformer':
            output = low_rank_model(data)
            output = output.view(-1, ntokens)
        else:
            hidden = repackage_hidden(hidden)
            output, hidden = low_rank_model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(low_rank_model.parameters(), args.clip)
        for p in low_rank_model.parameters():
           p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            logger.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        if args.dry_run:
            break


def train_vanilla():
    # Turn on training mode which enables dropout.
    vanilla_model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = vanilla_model.init_hidden(args.batch_size)

    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        vanilla_model.zero_grad()

        if args.model == 'Transformer':
            output = vanilla_model(data)
            output = output.view(-1, ntokens)
        else:
            hidden = repackage_hidden(hidden)
            output, hidden = vanilla_model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(vanilla_model.parameters(), args.clip)
        for p in vanilla_model.parameters():
           p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            logger.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        if args.dry_run:
            break


def export_onnx(path, batch_size, seq_len):
    logger.info('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


# Loop over epochs.
lr = args.lr
best_val_loss = None
logger.info("==> vanilla model: {}".format(vanilla_model))
logger.info("==> low rank model: {}".format(low_rank_model))
#optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
#scheduler = StepLR(optimizer=optimizer, step_size=1, gamma=0.9)

if args.evaluate:
    # Load the best saved model.
    if args.warmup_epoch < args.epochs:
        # pufferfish training mode
        saved_path = "pufferfish_model_lowrank_seed{}_best.pt".format(args.seed)
    else:
        # vanilla training mode
        saved_path = "vanilla_model_seed{}_best.pt".format(args.seed)

    with open(saved_path, 'rb') as f:
        if args.warmup_epoch < args.epochs:
            low_rank_model = torch.load(f)
        else:
            vanilla_model = torch.load(f)
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        # Currently, only rnn model supports flatten_parameters function.
        #if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
        #    model.rnn.flatten_parameters()

    # Run on test data.
    if args.warmup_epoch < args.epochs:
        # evaluate pufferfish model
        test_loss = evaluate(test_data)
    else:
        # evaluate vanilla model
        test_loss = evaluate_vanilla(test_data)

    logger.info('=' * 89)
    logger.info('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    logger.info('=' * 89)
    exit()

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(0, args.epochs):
        epoch_start_time = time.time()
        if args.warmup and epoch in range(args.warmup_epoch):
            logger.info("#### Full-rank Warming up, epoch: {}".format(epoch))
            train_vanilla()
        elif args.warmup and epoch == args.warmup_epoch:
            logger.info("#### Switching to low rank, epoch: {}".format(epoch))

            decompose_start = torch.cuda.Event(enable_timing=True)
            decompose_end = torch.cuda.Event(enable_timing=True)

            decompose_start.record()
            low_rank_model = decompose_vanilla_model(vanilla_model=vanilla_model, 
                                        low_rank_model=low_rank_model, 
                                        rank_ratio=0.25)
            decompose_end.record()
            torch.cuda.synchronize()
            decompose_dur = float(decompose_start.elapsed_time(decompose_end))/1000.0
            logger.info("#### Cost for decomposing the weights: {} ....".format(decompose_dur))
            
            lr /= 2.0
            train()
        else:
            logger.info("#### Low-rank Training, epoch: {}".format(epoch))
            train()

        if args.warmup and epoch in range(args.warmup_epoch):
            val_loss = evaluate_vanilla(val_data)
        else:
            val_loss = evaluate(val_data)
        logger.info('-' * 89)
        logger.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        logger.info('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            #with open(args.save, 'wb') as f:
            #    torch.save(vanilla_model, f)
            if args.warmup:
                if epoch in range(args.warmup_epoch):
                    with open("vanilla_model_seed{}_best.pt".format(args.seed), 'wb') as f:
                        torch.save(vanilla_model, f)
                else:
                    with open("pufferfish_model_lowrank_seed{}_best.pt".format(args.seed), 'wb') as f:
                        torch.save(low_rank_model, f)                                  

            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
        #scheduler.step()

except KeyboardInterrupt:
    logger.info('-' * 89)
    logger.info('Exiting from training early')

# Load the best saved model.
if args.warmup_epoch < args.epochs:
    # pufferfish training mode
    saved_path = "pufferfish_model_lowrank_seed{}_best.pt".format(args.seed)
else:
    # vanilla training mode
    saved_path = "vanilla_model_seed{}_best.pt".format(args.seed)

with open(saved_path, 'rb') as f:
    if args.warmup_epoch < args.epochs:
        low_rank_model = torch.load(f)
    else:
        vanilla_model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    # Currently, only rnn model supports flatten_parameters function.
    #if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
    #    model.rnn.flatten_parameters()

# Run on test data.
if args.warmup_epoch < args.epochs:
    # evaluate pufferfish model
    test_loss = evaluate(test_data)
else:
    # evaluate vanilla model
    test_loss = evaluate_vanilla(test_data)

logger.info('=' * 89)
logger.info('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
logger.info('=' * 89)
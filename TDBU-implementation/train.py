import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim


from utils import utils

from model.TDBU import TDBU


def train(train_data, model, optimizer, criterion, log=True):
    model.train()
    train_loss = 0.0
    total_seen = 0
    correct = 0.0
    for batch_idx, inputs in enumerate(train_data):
        inputs = inputs.float().to(device)
        seq_len = inputs.shape[-1]//2
        X = inputs[:, :seq_len]
        Y = inputs[:, seq_len:]

        model.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, Y)
        loss = loss.mean()
        loss.backward()
        optimizer.step()

        predictions = torch.clip(outputs, 0, 1)
        predictions = (predictions > 0.5).float()
        total_seen += Y.size(0)
        train_loss += loss.item()
        correct += predictions.eq(Y).sum().item()

    accuracy = 100.*correct/(seq_len*total_seen)
    if log:
        print('Train Loss: %.3f | Train Acc: %.3f' %
              (train_loss/(batch_idx+1), accuracy))

    return accuracy

def test(model, test_data, log=True):
    model.eval()
    correct = 0.0
    predictions = None
    for batch_idx, inputs in enumerate(test_data):
        inputs = inputs.float().to(device)
        seq_len = inputs.shape[-1]//2
        X = inputs[:, :seq_len]
        Y = inputs[:, seq_len:]
        outputs = model(X)
        predictions = torch.clip(outputs, 0, 1)
        predictions = (predictions > 0.5).float()
        correct += torch.prod(predictions.eq(Y).float(), dim=1).item()

    if log:
        print('Test Acc: %.3f' % (100.*correct/len(test_data)))
    return outputs.detach(), correct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_step', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--n1', type=int, default=6)
    parser.add_argument('--n2', type=int, default=6)
    parser.add_argument('--n3', type=int, default=6)
    parser.add_argument('--n4', type=int, default=6)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--filter_size', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--output_dir', type=str, default='./output')
    opt = parser.parse_args()

    device = torch.device('cpu' if opt.no_cuda else 'cuda')

    if not os.path.exists(opt.output_dir + '/last/models'):
        os.makedirs(opt.output_dir + '/last/models')
    if not os.path.exists(opt.data_dir):
        os.makedirs(opt.data_dir)

    train_data, test_data, i_vocab_size, t_vocab_size = utils.load_data()
    if i_vocab_size is not None:
        print("# of vocabs (input):", i_vocab_size)
    print("# of vocabs (target):", t_vocab_size)

    
    model = TDBU(i_vocab_size, t_vocab_size,
                        n_layers=opt.n_layers,
                        hidden_size=opt.hidden_size,
                        filter_size=opt.filter_size,
                        dropout_rate=opt.dropout,
                        share_target_embedding=opt.share_target_embedding,
                        has_inputs=opt.has_inputs,
                        src_pad_idx=opt.src_pad_idx,
                        trg_pad_idx=opt.trg_pad_idx)
    model = model.to(device=device)

    if opt.parallel:
        print("Use", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("# of parameters: {}".format(num_params))

    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.MSELoss(reduction="none")

    for t_step in range(opt.train_step):
        print("Epoch", t_step)
        start_epoch_time = time.time()
        train(train_data, model, optimizer, criterion)
        print("Epoch Time: {:.2f} sec".format(time.time() - start_epoch_time))

        if t_step % 5 != 0:
            continue

        test(model, test_data)


if __name__ == '__main__':
    main()
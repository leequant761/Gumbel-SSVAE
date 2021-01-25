import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from models import Model
from utils import MNISTCached, setup_data_loaders

def compute_loss(x, x_recon, z_q, y_q, z_q_dist, y_q_dist, model, args):
    if args.kld == 'eric':
        bce, kl = model.approximate_loss(x, x_recon, z_q, z_q_dist, y_q_dist)
    else:
        bce, kl = model.loss(x, x_recon, z_q, y_q, z_q_dist, y_q_dist)
    elbo = -bce - kl
    loss = -elbo
    return loss

def compute_class_loss(y, y_q_dist):
    target = y.max(dim=1)[1]
    return F.cross_entropy(y_q_dist.logits, target, reduction='sum')

def train(epoch, model, sup_iter, unsup_iter, num_batches, period, optimizer, device, args):
    model = model.train()

    train_loss = 0.
    class_loss = 0.
    total_num = 0
    for nth in range(1, num_batches+1):
        if nth % period == 0:
            x, y = next(sup_iter)
        else:
            x, y = next(unsup_iter)
            y = None
        x = x.to(device)
        n_sample = x.size(0)
        x = x.view(n_sample, -1)
        total_num += n_sample
        
        # scheduler for temperature
        if nth % args.temp_interval == 0:
            n_updates = epoch * num_batches + nth
            temp = max(torch.tensor(args.init_temp) * np.exp(-n_updates*args.temp_anneal), torch.tensor(args.min_temp))
            model.temp = temp

        # compute & optimize the loss function
        x_recon, z_q, y_q, z_q_dist, y_q_dist = model(x, y)
        loss = compute_loss(x, x_recon, z_q, y_q, z_q_dist, y_q_dist, model, args)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # additional classification loss
        if y is not None:
            _, _, _, _, y_q_dist = model(x, y)
            loss_c = compute_class_loss(y, y_q_dist)
            optimizer.zero_grad()
            loss_c.backward()
            optimizer.step()
            class_loss += loss_c.item()

    # report results
    print('====> Epoch: {} Average negative ELBO: {:.4f}'.format(
            epoch, train_loss / total_num))
    print('====> Epoch: {} Average Classification loss: {:.4f}'.format(
            epoch, class_loss /  total_num * period))

def test(epoch, model, test_loader, device, args):
    model.eval()

    test_loss = 0
    is_observed = True
    correct_num = 0
    with torch.no_grad():
        for nth, (x, y) in enumerate(test_loader):
            x = x.to(device)
            x = x.view(x.size(0), -1)

            # compute the loss function
            x_recon, z_q, y_q, z_q_dist, y_q_dist = model(x, y)
            loss = compute_loss(x, x_recon, z_q, y_q, z_q_dist, y_q_dist, model, is_observed, args)
            loss += compute_class_loss(y, y_q_dist)
            test_loss += loss

            # compute accuracy
            y_pred = y_q_dist.probs
            correct_num += sum(torch.argmax(y, axis=1) == torch.argmax(y_pred, axis=1))

            # save reconstructed figure
            if nth == 0:
                n = min(x.size(0), 8)
                comparison = torch.cat([x[:n].reshape(-1, 1, 28, 28),
                                        x_recon.view(args.batch_size, 1, 28, 28)[:n]])
                if 'results' not in os.listdir():
                    os.mkdir('results')
                save_image(comparison.cpu(),
                            'results/reconstruction_' + str(epoch) + '.png', nrow=n)
    # report results
    total_num = len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss / total_num))
    print('====> Test set accuracy: {:.4f}'.format(correct_num.item() / total_num))

def main(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")

    data_loaders = setup_data_loaders(
                                    dataset=MNISTCached,
                                    use_cuda=True,
                                    batch_size=args.batch_size,
                                    sup_num=100,
                                    root='./.data'
                                    )
    train_sup_loader = data_loaders['sup']
    train_unsup_loader = data_loaders['unsup']
    test_loader = data_loaders['test']

    model = Model(args.init_temp, args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_batches = len(train_sup_loader) + len(train_unsup_loader)
    period = num_batches // len(train_sup_loader)
    for epoch in range(1, args.epochs + 1):
        sup_iter = iter(train_sup_loader)
        unsup_iter = iter(train_unsup_loader)
        train(epoch, model, sup_iter, unsup_iter, num_batches, period, optimizer, device, args)
        test(epoch, model, test_loader, device, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='.data/',
                        help='where is you mnist?')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--latent-dim', type=int, default=50, metavar='N',
                        help='the dimension for z_style latent variables')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--init-temp', type=float, default=1.0)
    parser.add_argument('--temp-anneal', type=float, default=0.00009)
    parser.add_argument('--temp-interval', type=float, default=300)
    parser.add_argument('--min-temp', type=float, default=0.1)

    parser.add_argument('--sampling', type=str, default='TDModel',
                        help='example: TDModel utilizes torch.distributions.relaxed, ExpTDModel stabilizes loss function')
    parser.add_argument('--kld', type=str, default='eric',
                        help='example: eric, madisson')

    args = parser.parse_args()
    main(args)
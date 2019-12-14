import torch
import torchvision

from modules.supernetwork import Supernetwork

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import robust_loss_pytorch.general

from radam import RAdam

adaptive = robust_loss_pytorch.adaptive.AdaptiveLossFunction(
    num_dims = 1, float_dtype=np.float32, device='cuda:0')

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    counter = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, router_logit = model(data)
        batch_size = target.size(0)
        # NOTE: this is a trap.
        target = target.unsqueeze(0).expand(2, -1).t().reshape(-1)
        scores = model.last_result.trajectory_scores.view(
            batch_size * 2, -1).clone()
        loss = F.nll_loss(output, target, reduction='none') # + scores.squeeze())
        loss = (loss * router_logit.t().reshape(-1)) / router_logit.t().reshape(-1).detach()
        loss = torch.mean(loss)
        loss.backward()
        optimizer.step()
        # print(loss.mean().item(), kl_loss.mean().item())
        if batch_idx % args.log_interval == 0:
            # print(model.last_result.trajectories.transpose(0, 1))
            # print(model.last_result.trajectory_scores.squeeze())
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            batch_size = target.size(0)
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            # argmax_result = model.last_result.trajectory_scores.squeeze().argmin(1)
            # output = output[argmax_result]
            target = target.unsqueeze(0).expand(2, -1).t().reshape(-1)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset) * 2,
        100. * correct / (len(test_loader.dataset) * 2)))# )))

    import time
    time.sleep(0.5)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./datasets', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Resize(32),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./datasets', train=False, transform=transforms.Compose([
                           transforms.Resize(32),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = Supernetwork(1, 1).to(device)
    optimizer = optim.Adam(
        model.parameters(),
    )

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")


if __name__ == '__main__':
    main()

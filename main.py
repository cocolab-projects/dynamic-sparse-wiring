# Based on github.com/ermongroup/subsets
import torch


EPSILON = torch.finfo(torch.float32).tiny


def subset_operator(scores, k, tau=1.0, hard=False):
    m = torch.distributions.gumbel.Gumbel(torch.zeros_like(scores),
                                          torch.ones_like(scores))
    g = m.sample()
    scores = scores + g

    khot = torch.zeros_like(scores)
    onehot_approx = torch.zeros_like(scores)

    for _ in range(k):
        khot_mask = torch.max(1.0 - onehot_approx, torch.full_like(scores, EPSILON))
        scores = scores + torch.log(khot_mask)
        onehot_approx = torch.nn.functional.softmax(scores / tau, dim=1)
        khot = khot + onehot_approx

    if hard:
        khot_hard = torch.zeros_like(khot)
        val, ind = torch.topk(khot, k, dim=1)
        # switched scatter from index_add_ to support batching
        khot = khot_hard.scatter(1, ind, 1) - khot.detach() + khot

    return khot


if __name__ == '__main__':
    examples = torch.nn.functional.softmax(torch.randn(2, 5, requires_grad=True), dim=0)
    print(subset_operator(examples, k=2, hard=True))

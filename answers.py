r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    # raise NotImplementedError()
    wstd = 0.1
    lr = 5e-3
    reg = 1e-2
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 1e-2
    lr_vanilla = 3e-4
    lr_momentum = 1e-2
    lr_rmsprop = 3e-4
    reg = 1e-5
    # ========================
    return dict(wstd=wstd, lr_vanilla=lr_vanilla, lr_momentum=lr_momentum,
                lr_rmsprop=lr_rmsprop, reg=reg)


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 1e-2
    lr = 1e-3
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
1. Without dropout the model converges faster but at some point (epoch 9-10) it starts to overfit while with dropout it converges slowlier and overfits less.
2. With high dropout the model almost doesn't train because too many neurons are dropped out and neurons don't manage to become meaningful combinations
which correspond to some features.
"""

part2_q2 = r"""
It is possible if scores for right classes predicted correctly slightly decrease (entropy becomes higher but accuracy stays the same),
scores for border-line incorrect predictions become correct while they lost slightly to incorrect class (entropy becomes a bit less
but accuracy becomes higher), scores for incorrect predictions become higher which does not decrease accuracy but increses entropy.
If it happens so that there is a combination of these phenomena, we can see a situation of slightly increasing
entropy and at the same time increasing accuracy.
"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
1. The best result for K=[32] was achieved for L=2 and the best results for K=[64] was achieved for
L=4, but the results were close for L=2 and L=4. It is likely that the best configuration is L=4. It is
so because it covers majority of types of complex spatial features.
2. The network became untrainable for L=8 and L=16. It likely happened because of exploding gradient
problem in a deep network. This could be solved by using different activation function.
"""

part3_q2 = r"""
The results for L=2 were slightly better for K=[128] but were close to K=[256],
for L=4 they were better for K=[256] because this way network could capture more different spatial features.
The same way as in Q1 network became untrainable for L=8.
"""

part3_q3 = r"""
For L=1 and L=2 model trained to approximately the same results, the model with L=1 trained significantly faster.
Fot L=3 and L=4 the model suffered from the same problem as in Q1 and Q2.
"""


part3_q4 = r"""
1. We added dropout to classifier layer for regularization, and the first convolutional layer has 5x5 kernels to capture
more complicated features at the first layer.
2. For L=2,3,4 the networks suffer from the same problem of exploding gradients as in Q1-Q3.
For L=1 the network has higher accuracy than networks in experiment 1.
"""
# ==============

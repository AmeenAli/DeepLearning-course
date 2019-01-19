import abc
import torch
from torch import Tensor


class Optimizer(abc.ABC):
    """
    Base class for optimizers.
    """
    def __init__(self, params):
        """
        :param params: A sequence of model parameters to optimize. Can be a
        list of (param,grad) tuples as returned by the Blocks, or a list of
        pytorch tensors in which case the grad will be taken from them.
        """
        assert isinstance(params, list) or isinstance(params, tuple)
        self._params = params

    @property
    def params(self):
        """
        :return: A sequence of parameter tuples, each tuple containing
        (param_data, param_grad). The data should be updated in-place
        according to the grad.
        """
        returned_params = []
        for x in self._params:
            if isinstance(x, Tensor):
                p = x.data
                dp = x.grad.data if x.grad is not None else None
                returned_params.append((p, dp))
            elif isinstance(x, tuple) and len(x) == 2:
                returned_params.append(x)
            else:
                raise TypeError(f"Unexpected parameter type for parameter {x}")

        return returned_params

    def zero_grad(self):
        """
        Sets the gradient of the optimized parameters to zero (in place).
        """
        for p, dp in self.params:
            dp.zero_()

    @abc.abstractmethod
    def step(self):
        """
        Updates all the registered parameter values based on their gradients.
        """
        raise NotImplementedError()


class VanillaSGD(Optimizer):
    def __init__(self, params, learn_rate=1e-3, reg=0):
        """
        :param params: The model parameters to optimize
        :param learn_rate: Learning rate
        :param reg: L2 Regularization strength
        """
        super().__init__(params)
        self.learn_rate = learn_rate
        self.reg = reg

    def step(self):
        for p, dp in self.params:
            if dp is None:
                continue
            p -= (dp + self.reg * p) * self.learn_rate



class MomentumSGD(Optimizer):
    def __init__(self, params, learn_rate=1e-3, reg=0, momentum=0.9):
        """
        :param params: The model parameters to optimize
        :param learn_rate: Learning rate
        :param reg: L2 Regularization strength
        :param momentum: Momentum factor
        """
        super().__init__(params)
        self.learn_rate = learn_rate
        self.reg = reg
        self.momentum = momentum

        self.velocities = []
        for p, dp in params:
            velocity = None
            if dp is not None:
                velocity = torch.zeros_like(dp)
            self.velocities.append(velocity)


    def step(self):
        for i, (p, dp) in enumerate(self.params):
            if dp is None:
                continue
            self.velocities[i] = (self.velocities[i] * self.momentum -
                                  (dp + self.reg * p) * self.learn_rate)
            p += self.velocities[i]



class RMSProp(Optimizer):
    def __init__(self, params, learn_rate=1e-3, reg=0, decay=0.99, eps=1e-8):
        """
        :param params: The model parameters to optimize
        :param learn_rate: Learning rate
        :param reg: L2 Regularization strength
        :param decay: Gradient exponential decay factor
        :param eps: Constant to add to gradient sum for numerical stability
        """
        super().__init__(params)
        self.learn_rate = learn_rate
        self.reg = reg
        self.decay = decay
        self.eps = eps

        self.rms = []
        for p, dp in params:
            rms = None
            if dp is not None:
                rms = torch.zeros_like(dp)
            self.rms.append(rms)


    def step(self):
        for i, (p, dp) in enumerate(self.params):
            if dp is None:
                continue

            cgrad = dp + self.reg * p
            self.rms[i] = self.decay * self.rms[i] + (1 - self.decay) * (cgrad ** 2)
            p -= self.learn_rate * (cgrad / ((self.rms[i] + self.eps) ** 0.5))


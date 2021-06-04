from torch.autograd import Function


class GradientReversalLayer(Function):
    r"""Gradient reversal layer as suggested by Ganin et al. https://arxiv.org/abs/1505.07818 
    """
    @staticmethod
    def forward(ctx, x, alpha):
        # store alpha to object context
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # invert gradient with dampped alpha value
        output = grad_output.neg() * ctx.alpha
        return output, None

"""Hyperbolic operations utils functions."""

import torch

MIN_NORM = 1e-15
BALL_EPS = {torch.float32: 4e-3, torch.float64: 1e-5}

# ################# MATH FUNCTIONS ########################

class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(x)
        dtype = x.dtype
        x = x.double()
        return (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5).to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


def artanh(x):
    return Artanh.apply(x)


def tanh(x):
    return x.clamp(-15, 15).tanh()


# ################# HYPERBOLIC FUNCTION WITH CURVATURES=1 ########################


def givens_rotations(r, x):
    """Givens rotations.

    Args:
        r: torch.Tensor of shape (N x d), rotation parameters
        x: torch.Tensor of shape (N x d), points to rotate

    Returns:
        torch.Tensor os shape (N x d) representing rotation of x by r
    """
    # [batch_size, dim // 2, 2]
    givens = r.view((r.shape[0], -1, 2))
    # L2 norm to constrain sine/cosine matrix
    givens = givens / torch.norm(givens, p=2, dim=-1,
                                 keepdim=True).clamp_min(1e-15)
    # [batch_size, dim // 2, 2]
    x = x.view((r.shape[0], -1, 2))
    # cosine: givens[:, :, 0:1] (batch_size, dim//2, 1)
    # sine: givens[:, :, 1:]  (batch_size, dim//2, 1)
    x_rot = givens[:, :, 0:1] * x + givens[:, :, 1:] * \
            torch.cat((-x[:, :, 1:], x[:, :, 0:1]), dim=-1)
    return x_rot.view((r.shape[0], -1))


def givens_reflection(r, x):
    """Givens reflections.

    Args:
        r: torch.Tensor of shape (N x d), rotation parameters
        x: torch.Tensor of shape (N x d), points to reflect

    Returns:
        torch.Tensor os shape (N x d) representing reflection of x by r
    """
    # [batch_size, dim // 2, 2]
    givens = r.view((r.shape[0], -1, 2))
    # L2 norm to constrain sine/cosine matrix
    givens = givens / torch.norm(givens, p=2, dim=-1,
                                 keepdim=True).clamp_min(1e-15)
    # [batch_size, dim // 2, 2]
    x = x.view((r.shape[0], -1, 2))
    # cosine: givens[:, :, 0:1] (batch_size, dim//2, 1)
    # sine: givens[:, :, 1:]  (batch_size, dim//2, 1)
    x_ref = givens[:, :, 0:1] * torch.cat((x[:, :, 0:1], -x[:, :, 1:]), dim=-1) + givens[:, :, 1:] * torch.cat(
        (x[:, :, 1:], x[:, :, 0:1]), dim=-1)
    return x_ref.view((r.shape[0], -1))


# ################# constant curvature=1 ########################


def _lambda_x(x, c=1):
    x_sqnorm = torch.sum(x.data.pow(2), dim=-1, keepdim=True)
    return 2 / (1. - c * x_sqnorm).clamp_min(MIN_NORM)


def expmap(u, base):
    """Exponential map u in the tangent space of point base with curvature c.
        from NIPS18 Hyperbolic Neural Networks

    Args:
        u: torch.Tensor of size B x d with tangent points
        base: torch.Tensor of size B x d with hyperbolic points

    Returns:
        torch.Tensor with  hyperbolic points.
    """
    # p is in hyperbolic space
    u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    second_term = (
            tanh(1 / 2 * _lambda_x(base) * u_norm)
            * u
            / (u_norm)
    )
    gamma_1 = mobius_add(base, second_term)
    return gamma_1


def expmap0(u):
    """Exponential map taken at the origin of the Poincare ball with curvature c.

    Args:
        u: torch.Tensor of size B x d with tangent points

    Returns:
        torch.Tensor with  hyperbolic points.
    """
    # see equation 1 for detail
    u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    gamma_1 = tanh(u_norm) * u / (u_norm)
    return project(gamma_1)


def logmap(y, base):
    """Logarithmic map taken at the x of the Poincare ball with curvature c.
       from NIPS18 Hyperbolic Neural Networks

    Args:
        y: torch.Tensor of size B x d with hyperbolic points
        base: torch.Tensor of size B x d with hyperbolic points

    Returns:
        torch.Tensor with tangent points.
    """

    sub = mobius_add(-base, y)
    sub_norm = sub.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    lam = _lambda_x(base)
    return 2 / lam * artanh(sub_norm) * sub / sub_norm


def logmap0(y):
    """Logarithmic map taken at the origin of the Poincare ball with curvature c.

    Args:
        y: torch.Tensor of size B x d with hyperbolic points

    Returns:
        torch.Tensor with tangent points.
    """
    y_norm = y.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    return y / y_norm / artanh(y_norm)


def project(x):
    """Project points to Poincare ball with curvature c.
    Need to make sure hyperbolic embeddings are inside the unit ball.

    Args:
        x: torch.Tensor of size B x d with hyperbolic points
    Returns:
        torch.Tensor with projected hyperbolic points.
    """
    # [batch_size, dim]
    norm = x.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    eps = BALL_EPS[x.dtype]
    maxnorm = (1 - eps)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)


def mobius_add(x, y):
    """Mobius addition of points in the Poincare ball with curvature c.

    Args:
        x: torch.Tensor of size B x d with hyperbolic points
        y: torch.Tensor of size B x d with hyperbolic points

    Returns:
        Tensor of shape B x d representing the element-wise Mobius addition of x and y.
    """
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    xy = torch.sum(x * y, dim=-1, keepdim=True)
    num = (1 + 2 * xy + y2) * x + (1 - x2) * y
    denom = 1 + 2 * xy + x2 * y2
    return num / denom.clamp_min(MIN_NORM)


def hyp_distance(x, y):
    """Hyperbolic distance on the Poincare ball with curvature c.

    Args:
        x: torch.Tensor of size B x d with hyperbolic queries
        y: torch.Tensor with hyperbolic queries, shape n_entities x d if eval_mode is true else (B x d)

    Returns: torch,Tensor with hyperbolic distances, size B x 1 if eval_mode is False
            else B x n_entities matrix with all pairs distances
    """
    dist_c = artanh(mobius_add(-x, y).norm(dim=-1, p=2, keepdim=False))
    dist = dist_c * 2
    return dist


def sq_hyp_distance(x, y):
    """Square Hyperbolic distance on the Poincare ball with curvature c.

    Args:
        x: torch.Tensor of size B x d with hyperbolic queries
        y: torch.Tensor with hyperbolic queries, shape n_entities x d if eval_mode is true else (B x d)

    Returns: torch,Tensor with hyperbolic distances, size B x 1 if eval_mode is False
            else B x n_entities matrix with all pairs distances
    """
    dist_c = artanh(mobius_add(-x, y).norm(dim=-1, p=2, keepdim=False))
    dist = dist_c * 2
    return dist ** 2

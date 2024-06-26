DEBUG = True

# if true, no discrete generators will be searched for
ONLY_IDENTITY_COMPONENT = True

# how much we enforce that the \mu(g*x) = \mu(x) (without this, there may be no relation between \mu(gx) and \mu(x))
INVARIANCE_LOSS_COEFF = 3

# regularization that discovered group is not just the identity
IDENTITY_COLLAPSE_REGULARIZATION = 5e-1

# boolean for CUDA
DISABLE_CUDA = False


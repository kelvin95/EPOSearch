from .epo import EPO
from .graddrop import GradDrop
from .graddrop_random import GradDropRandom
from .graddrop_deterministic import GradDropDeterministic
from .gradnorm import GradNorm
from .gradvacc import GradVacc
from .gradortho import GradOrtho
from .gradalign import GradAlign
from .individual import Individual
from .itmtl import ITMTL
from .linscalar import LinScalar
from .meta import MetaLearner
from .mgda import MGDA
from .pcgrad import PCGrad
from .pmtl import PMTL

SOLVER_FACTORY = {
    "epo": EPO,
    "graddrop": GradDrop,
    "graddrop_random": GradDropRandom,
    "graddrop_deterministic": GradDropDeterministic,
    "gradnorm": GradNorm,
    "gradvacc": GradVacc,
    "gradortho": GradOrtho,
    "gradalign": GradAlign,
    "individual": Individual,
    "itmtl": ITMTL,
    "linscalar": LinScalar,
    "meta": MetaLearner,
    "mgda": MGDA,
    "pcgrad": PCGrad,
    "pmtl": PMTL,
}

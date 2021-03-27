from .epo import EPO
from .graddrop import GradDrop
from .graddrop_random import GradDropRandom
from .graddrop_deterministic import GradDropDeterministic
from .gradnorm import GradNorm
from .individual import Individual
from .itmtl import ITMTL
from .linscalar import LinScalar
from .mgda import MGDA
from .pcgrad import PCGrad
from .pmtl import PMTL

SOLVER_FACTORY = {
    "epo": EPO,
    "graddrop": GradDrop,
    "graddrop_random": GradDropRandom,
    "graddrop_deterministic": GradDropDeterministic,
    "gradnorm": GradNorm,
    "individual": Individual,
    "itmtl": ITMTL,
    "linscalar": LinScalar,
    "mgda": MGDA,
    "pcgrad": PCGrad,
    "pmtl": PMTL,
}

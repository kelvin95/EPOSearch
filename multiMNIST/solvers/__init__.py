from .epo import EPO
from .graddrop import GradDrop
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
    "gradnorm": GradNorm,
    "individual": Individual,
    "itmtl": ITMTL,
    "linscalar": LinScalar,
    "mgda": MGDA,
    "pcgrad": PCGrad,
    "pmtl": PMTL,
}
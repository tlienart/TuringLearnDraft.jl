# NOTE all of this should be done through specialised lib

gradient(::ZeroLoss) = x->0.0

gradient(::L2Loss) = x->x
gradient(lp::LpDiffLoss) = x->x.^(lp.p-1)

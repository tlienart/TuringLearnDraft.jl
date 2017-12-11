# NOTE all of this should be done through extran library (Lossfunctions)

loss(::ZeroLoss) = x->0.0

loss(::L1Loss) = x->sum(abs.(x))
loss(::L2Loss) = x->sum(x.^2)/2
loss(lp::Union{LpLoss, LpDiffLoss}) = x->sum(x.^lp.p)/p
loss(::LinfLoss) = x->maximum(x)

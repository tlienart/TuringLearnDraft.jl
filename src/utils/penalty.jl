export
    NoPenalty,
    penalty_coef,
    L1Penalty,
    L2Penalty,
    LPPenalty

struct NoPenalty <: Loss end

const L1Penalty = LossFunctions.ScaledDistanceLoss{L1DistLoss}
const L2Penalty = LossFunctions.ScaledDistanceLoss{L2DistLoss}
const LPPenalty{P} = LossFunctions.ScaledDistanceLoss{LPDistLoss{P}}

L1Penalty(λ::Real) = scaled(L1DistLoss(), Val{λ})
L2Penalty(λ::Real) = scaled(L2DistLoss(), Val{λ})
LPPenalty(p::Real, λ::Real) = scaled(LPDistLoss(p), Val{λ})

penalty_coef{T, K}(sl::LossFunctions.ScaledLoss{T, K}) = K

struct NoRegularizer <: Regularizer end

struct L1Regularizer <: Regularizer
    λ::AbstractFloat
end

mutable struct L2Regularizer <: Regularizer
    λ::AbstractFloat
end

const Lasso = L1Regularizer
const Ridge = L2Regularizer
const Tikhonov = L2Regularizer

export
    GeneralizedLinearRegression, GeneralizedLinearRegressionModel,
    LinearRegression,
    LassoRegression,
    RidgeRegression,
    ElasticNetRegression,
    LogisticRegression

## Declaration and Model types

mutable struct GeneralizedLinearRegression
    loss::DistanceLoss  # L(y-ŷ)
    penalty::Loss       # NOTE: the penalty contains the scaling
    fit_intercept::Bool
end

struct GeneralizedLinearRegressionModel <: RegressionModel
    glr::GeneralizedLinearRegression
    n_features::Int
    intercept::Real
    coefs::AbstractVector{Real}
end

### Constructors

function GeneralizedLinearRegression(;
    loss=L2DistLoss(),
    penalty=NoPenalty(),
    fit_intercept=true)

    GeneralizedLinearRegression(
        loss,
        penalty,
        fit_intercept)
end

function LinearRegression(;
    fit_intercept::Bool=true)

    GeneralizedLinearRegression(
        fit_intercept=fit_intercept)
end

function RidgeRegression(
    λ::Real=1.0;
    fit_intercept::Bool=true)

    GeneralizedLinearRegression(
        penalty=L2Penalty(λ),
        fit_intercept=fit_intercept)
end

function LassoRegression(
    λ::Real=1.0;
    fit_intercept::Bool=true)

    GeneralizedLinearRegression(
        penalty=L1Penalty(λ),
        fit_intercept=fit_intercept)
end

function HuberRegression(
    λ::Real=1.0,
    fit_intercept::Bool=true)

    GeneralizedLinearRegression(
        loss=HuberPenalty(λ),
        fit_intercept=fit_intercept)
end

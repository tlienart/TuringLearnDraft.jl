export
    GeneralizedLinearRegression, GeneralizedLinearRegressionModel,
    LinearRegression,
    LassoRegression,
    RidgeRegression,
    ElasticNetRegression,
    LogisticRegression

## Declaration and Model types

mutable struct GeneralizedLinearRegression
    loss::Loss
    regularizer::Loss
    regularizer_coef::AbstractFloat
    fit_intercept::Bool
    solver::String
end

struct GeneralizedLinearRegressionModel <: RegressionModel
    glr::GeneralizedLinearRegression
    dim::Tuple{Int, Int}
    intercept::AbstractFloat
    coefs::AbstractArray{AbstractFloat}
end

### Constructors

function GeneralizedLinearRegression(;
    loss::Loss=L2Loss(),
    regularizer::Loss=ZeroLoss(),
    regularizer_coef::AbstractFloat=0.0,
    fit_intercept::Bool=true,
    solver::String="default")

    GeneralizedLinearRegression(
        loss,
        regularizer,
        regularizer_coef,
        fit_intercept,
        solver)
end

function LinearRegression(;
    fit_intercept::Bool=true,
    solver::String="default")
    GeneralizedLinearRegression(
        fit_intercept=fit_intercept,
        solver=solver)
end

function RidgeRegression(
    λ::AbstractFloat=1.0;
    fit_intercept::Bool=true,
    solver::String="default")

    GeneralizedLinearRegression(
        regularizer=L2Loss(),
        regularizer_coef=λ,
        fit_intercept=fit_intercept,
        solver=solver)
end

function LassoRegression(
    λ::AbstractFloat=1.0;
    fit_intercept::Bool=true,
    solver::String="default")

    GeneralizedLinearRegression(
        regularizer=L1Loss(),
        regularizer_coef=λ,
        fit_intercept=fit_intercept,
        solver=solver)
end

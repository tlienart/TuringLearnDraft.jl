export
    LinearRegression,
    RidgeRegression,
    LassoRegression,
    LinearRegressionModel,
    LogisticRegression,
    LogisticRegressionModel

mutable struct LinearRegression
    ρ::Regularizer
end
LinearRegression() = LinearRegression(NoRegularizer())
RidgeRegression(λ::AbstractFloat) = LinearRegression(Ridge(λ))
LassoRegression(λ::AbstractFloat) = LinearRegression(Lasso(λ))

struct LinearRegressionModel <: RegressionModel
    lr::LinearRegression
    dim::Tuple{Int, Int} # number of instances, number of features
    β::AbstractArray{AbstractFloat}
end

mutable struct LogisticRegression
    ρ::Regularizer
end

struct LogisticRegressionModel <: RegressionModel
    lr::LogisticRegression
    β::AbstractArray{AbstractFloat}
end

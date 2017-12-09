function fit(
    lr::LinearRegression,
    X::AbstractArray{T},
    y::AbstractVector{T};
    solver::String="default") where T <: AbstractFloat

    if size(X, 1) != length(y)
        throw(DimensionMismatch("X has $(size(X, 1)) rows and y has" *
            "size $(size(y)) elements, they should be equal." ))
    end

    # Coefficient of the linear model
    β = AbstractArray{T, 1}

    if typeof(lr.ρ) == NoRegularizer
        if solver == "default"
            β = X \ y
        else
            # NOTE: here could have different ways of solving a linear system
            # whichever may be appropriate such as, for example, Krylov
            throw(UnimplementedException())
        end

    elseif typeof(lr.ρ) == Ridge
        if solver == "default"
            β = (X' * X + lr.ρ.λ * eye(size(X, 2))) \ (X' * y)
        else
            throw(UnimplementedException())
        end

    elseif typeof(lr.ρ) == Lasso
        throw(UnimplementedException())

    end

    LinearRegressionModel(lr, size(X), β)
end

function predict(
    lrm::LinearRegressionModel,
    X::AbstractArray{T}) where T <: AbstractFloat

    if lrm.dim[2] != size(X, 2)
        throw(DimensionMismatch("X has $(size(X, 2)) columns, the model" *
            "expected $(lrm.dim[2])." ))
    end

    X * lrm.β
end

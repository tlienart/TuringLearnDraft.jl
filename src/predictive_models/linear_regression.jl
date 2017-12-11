function predict(
    glrm::GeneralizedLinearRegressionModel,
    X::AbstractArray{T}) where T <: AbstractFloat

    if glrm.dim[2] != size(X, 2)
        throw(DimensionMismatch("X has $(size(X, 2)) columns, the model" *
            "expected $(glrm.dim[2])." ))
    end

    if glrm.glr.fit_intercept
        X * glrm.coefs .+ glrm.intercept
    else
        X * glrm.coefs
    end
end

function fit(
    glr::GeneralizedLinearRegression,
    X::AbstractArray{T},
    y::AbstractVector{T}) where T <: AbstractFloat

    if size(X, 1) != length(y)
        throw(DimensionMismatch("X has $(size(X, 1)) rows and y has" *
            "size $(size(y)) elements, they should be equal." ))
    end

    (n, p) = size(X)

    # Coefficient of the linear model
    β = AbstractArray{T, 1}

    if glr.fit_intercept
        # NOTE this is probably inefficient (copy)
        X = hcat(ones(n), X)
    end

    if typeof(glr.loss) == L2Loss

        typeof_regularizer = typeof(glr.regularizer)

        if typeof_regularizer == ZeroLoss
            β = olsfit(glr, X, y)
        elseif typeof_regularizer == L1Loss
            β = lassofit(glr, X_, y)
        elseif typeof_regularizer == L2Loss
            β = ridgefit(glr, X, y)
        else
            throw(UnimplementedException())
        end

    else
        throw(UnimplementedException())
    end

    intercept = glr.fit_intercept ? β[1] : zero(T)
    coefs = glr.fit_intercept ? β[2:end] : β

    GeneralizedLinearRegressionModel(
        glr,
        (n, p),
        intercept,
        coefs)
end


function olsfit(glr, X, y)
    if glr.solver == "default"
        X \ y
    else
        throw(UnimplementedException())
    end
end


function ridgefit(glr, X, y)
    if glr.solver == "default"
        (X' * X + glr.regularizer_coef * eye(size(X, 2))) \ (X' * y)
    else
        throw(UnimplementedException())
    end
end


function lassofit(glr, X, y)
    if glr.solver == "default"
        throw(UnimplementedException())
    else
        throw(UnimplementedException())
    end
end

function predict(
    glrm::GeneralizedLinearRegressionModel,
    X::AbstractArray{T}) where T <: Number

    if glrm.n_features != size(X, 2)
        throw(DimensionMismatch(
            "X has $(size(X, 2)) columns, the model" *
            "expected $(glrm.dim[2])." ))
    end

    if glrm.glr.fit_intercept
        X * glrm.coefs + glrm.intercept
    else
        X * glrm.coefs
    end
end

function fit(
    glr::GeneralizedLinearRegression,
    X::AbstractArray{T},
    y::AbstractVector{T},
    solver::String="default") where T <: Number

    if size(X, 1) != length(y)
        throw(DimensionMismatch(
            "X has $(size(X, 1)) rows and y has" *
            "size $(size(y)) elements, they should be equal." ))
    end

    β = AbstractVector{T}

    # Prepare the feature matrix (with or without intercept)
    # TODO: check, the hcat is likely inefficient, the copy maybe also

    X_ = glr.fit_intercept ? hcat(ones(size(X, 1)), copy(X)) : copy(X)

    # Depending on the loss and penalty, a different subfunction is applied
    # itself depending upon the solver that is provided

    if glr.loss isa L2DistLoss

        if glr.penalty isa NoPenalty
            β = olsfit(glr, X_, y, solver)
        elseif glr.penalty isa L1Penalty
            λ = penalty_coef(glr.penalty)
            β = lassofit(glr, X_, y, λ, solver)
        elseif glr.penalty isa L2Penalty
            λ = penalty_coef(glr.penalty)
            β = ridgefit(glr, X_, y, λ, solver)
        else
            throw(UnimplementedException())
        end

    else
        throw(UnimplementedException())
    end

    (intercept, coefs) = glr.fit_intercept ? (β[1], β[2:end]) : (zero(T), β)

    GeneralizedLinearRegressionModel(
        glr,
        size(X, 2),
        intercept,
        coefs)
end


function olsfit(glr, X, y, solver)
    if solver == "default"
        X \ y
    else
        throw(UnimplementedException())
    end
end


function ridgefit(glr, X, y, λ, solver)
    if solver == "default"
        (X' * X + λ * eye(size(X, 2))) \ (X' * y)
    else
        throw(UnimplementedException())
    end
end


function lassofit(glr, X, y, λ, solver)
    if solver == "default"
        throw(UnimplementedException())
    else
        throw(UnimplementedException())
    end
end

abstract type Loss end
abstract type DifferentiableLoss <: Loss end

struct ZeroLoss <: DifferentiableLoss end # used e.g. when

### Lp Losses

struct L1Loss <: Loss end
struct L2Loss <: DifferentiableLoss end
struct LpLoss <: Loss
    p::AbstractFloat
end
struct LpDiffLoss <: DifferentiableLoss
    p::AbstractFloat
end
struct LinfLoss <: Loss end

### Constructors, where needed

function LpRegularizer(p::AbstractFloat)
    @assert(p>0, "Parameter for LpRegularizer must be positive")
    if p < 1
        LpLoss(p)
    elseif p == 1
        L1Loss()
    elseif isinf(p)
        LinfLoss()
    else
        LpDiffLoss(p)
    end
end

# ### Binary losses
#
# struct LogitLoss <: DifferentiableLoss end
# struct ProbitLoss <: Loss end
#
# ### Multiclass losses
#
# struct MultinomialLoss <: DifferentiableLoss end

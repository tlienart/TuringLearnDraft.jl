module TuringLearnDraft

export fit, predict

struct UnimplementedException <: Exception end

include("types_transformer_models.jl")
include("types_predictive_models.jl")

include("utils/types_losses.jl")
include("utils/losses.jl")
include("utils/gradient_losses.jl")

include("api_predictive_models/api_linear_regression.jl")
include("api_predictive_models/api_tree_regression.jl")

include("predictive_models/linear_regression.jl")

end # module

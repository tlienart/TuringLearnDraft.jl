module TuringLearnDraft

using LearnBase
using LossFunctions

export fit, predict

struct UnimplementedException <: Exception end

include("utils/types_transformer_models.jl")
include("utils/types_predictive_models.jl")
include("utils/penalty.jl")

include("api_predictive_models/api_linear_regression.jl")
include("api_predictive_models/api_tree_regression.jl")

include("predictive_models/linear_regression.jl")

end # module

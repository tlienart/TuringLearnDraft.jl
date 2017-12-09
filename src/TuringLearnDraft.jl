module TuringLearnDraft

export fit, predict

struct UnimplementedException <: Exception end

include("types.jl")

include("predictive_models_api/continuous_regressors.jl")
include("predictive_models_api/tree_regressors.jl")
include("predictive_models_api/regularizers.jl")

include("predictive_models/linear_regression.jl")

end # module

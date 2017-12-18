abstract type TransformerModel end

abstract type Projection <: TransformerModel end # PCA, ICA, ...
abstract type Clustering <: TransformerModel end # KMeans, Hierarchical, DBSCan

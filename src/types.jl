abstract type TransformerModel end

abstract type Projection <: TransformerModel end # PCA, ICA, ...
abstract type Clustering <: TransformerModel end # KMeans, Hierarchical, DBSCan

abstract type PredictiveModel end

abstract type RegressionModel <: PredictiveModel end

abstract type ClassificationModel <: PredictiveModel end

abstract type EnsembleModel <: PredictiveModel end
abstract type EnsembleRegressor <: EnsembleModel end
abstract type EnsembleClassifier <: EnsembleModel end

abstract type StackingModel <: PredictiveModel end
abstract type StackingRegressor <: StackingModel end
abstract type StackingClassifier <: StackingModel end

abstract type Regularizer end

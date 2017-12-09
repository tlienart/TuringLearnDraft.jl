using TuringLearnDraft, Base.Test

srand(126)

n = 30
p = 3
X = randn(n, p)
y = randn(n)
λ = 0.5

n2 = 10
X2 = randn(n2, p)


lr_noreg = LinearRegression()
lr_lasso = LassoRegression(λ)
lr_ridge = RidgeRegression(λ)

@test typeof(lr_noreg.ρ) == TuringLearnDraft.NoRegularizer
@test typeof(lr_lasso.ρ) == TuringLearnDraft.Lasso
@test typeof(lr_ridge.ρ) == TuringLearnDraft.Ridge

@test lr_lasso.ρ.λ == λ
@test lr_ridge.ρ.λ == λ

lrm_noreg = fit(lr_noreg, X, y)
lrm_ridge = fit(lr_ridge, X, y)

β_noreg = (X\y)
β_ridge = (X'*X+λ*eye(p))\(X'*y)
@test lrm_noreg.β == β_noreg
@test lrm_ridge.β == β_ridge

@test predict(lrm_noreg, X2) == X2*β_noreg
@test predict(lrm_ridge, X2) == X2*β_ridge

#lrm_lasso = firt(lr_lasso, X, y)

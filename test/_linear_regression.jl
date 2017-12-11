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
lr_lasso = LassoRegression(λ, fit_intercept=false)
lr_ridge = RidgeRegression(λ, fit_intercept=false)

lrm_noreg = fit(lr_noreg, X, y)
lrm_ridge = fit(lr_ridge, X, y)

β_noreg = hcat(ones(n), X) \ y
β_ridge = (X'*X+λ*eye(p))\(X'*y)
@test lrm_noreg.intercept == β_noreg[1]
@test lrm_noreg.coefs == β_noreg[2:end]
@test lrm_ridge.coefs == β_ridge

@test predict(lrm_noreg, X2) == X2 * β_noreg[2:end] .+ β_noreg[1]
@test predict(lrm_ridge, X2) == X2 * β_ridge

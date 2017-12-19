# TuringLearnDraft

[![Build Status](https://travis-ci.org/tlienart/TuringLearnDraft.jl.svg?branch=master)](https://travis-ci.org/tlienart/TuringLearnDraft.jl)

[![Coverage Status](https://coveralls.io/repos/tlienart/TuringLearnDraft.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/tlienart/TuringLearnDraft.jl?branch=master)

[![codecov.io](http://codecov.io/github/tlienart/TuringLearnDraft.jl/coverage.svg?branch=master)](http://codecov.io/github/tlienart/TuringLearnDraft.jl?branch=master)

## Roadmap

### All

* Data Processing
    * Connection with Julia DB
    * (?) Connection with DataFrame
    * Feature Processing
        * Scalers
    * Missing value processing (via Dataframe or julia db?)

* Predictive model
    * Continuous Regression models
        * Standard Lin Reg
            * [x] analytical solver
            * [] krylov solver
        * Ridge Lin Reg
            * [x] analytical solver
        * Lasso Lin Reg
            * [] fista solver
        * Logistic reg
            * [] bfgs solver
        * Other reg
            * [LOESS reg](https://github.com/JuliaStats/Loess.jl)
            * Huber reg
            * Quantile reg
    * Continuous Classification models
        * Logreg
            * with L2
                * [] lbfgs solver
                * [] sag solver
            * with L1
        * SVM
        * kNN
    * Tree-based models
        * Tree regression
        * Tree classification

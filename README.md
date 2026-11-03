# Regression-Competition

# House Price Regression (Ames, Iowa)

Team: Amina Salamatolla, Dianna Mirtayeva
Kaggle competition: [Regression for house prices](https://www.kaggle.com/competitions/regression-for-house-prices)

## Goal

Predict `SalePrice` for the 260 houses in the test set using the Ames housing data. Kaggle scores submissions with root mean squared error (RMSE) in dollars.

## Repository contents

| File | Description |
|---|---|
| `house_price_regression.ipynb` | Full workflow: EDA, cleaning, feature engineering, model comparison, final model and submission |
| `submission.csv` | Predictions uploaded to Kaggle (`Id`, `SalePrice`) |
| `README.md` | This file |

The data is not included in the repo (it is subject to the competition rules). To run the notebook, download `train.csv` and `test.csv` from the competition's Data tab into a `data/` folder. On Kaggle the notebook finds the files under `/kaggle/input` automatically.

## Approach

1. **EDA.** `SalePrice` is right-skewed (skew 1.97), so models are trained on log(SalePrice) and predictions are converted back to dollars. `OverallQual` and `GrLivArea` are the strongest predictors.
2. **Cleaning.** For most columns, NA means the house doesn't have that feature (no pool, no garage, no alley), so these are filled with "None" or 0. Remaining gaps (`LotFrontage`, one `Electrical` value) are imputed inside the pipeline.
3. **Features.** Quality ratings are mapped to ordered numbers. New features: total square footage, total bathrooms, house age and years since remodel at sale, total porch area, and quality × living area.
4. **Modeling.** All preprocessing sits in a scikit-learn pipeline so it is fit only on training folds. Models were compared with 5-fold cross-validated RMSE.
5. **Final model.** A weighted blend of Ridge regression (log target, log-transformed skewed features), gradient boosting (log target, tuned with a small grid search) and gradient boosting with Huber loss. Weights (0.5 / 0.2 / 0.3) were chosen by comparing out-of-fold RMSE for every combination in steps of 0.1.
6. **Out-of-range houses.** Ridge extrapolates for houses larger than anything in the training data (it predicted over $1M for one test house). For houses whose living area or square footage exceeds the training maximum, we use the gradient boosting prediction only. This affects one test house and leaves CV almost unchanged.

## Results (5-fold CV RMSE)

| Model | CV RMSE |
|---|---|
| Mean baseline | $80,817 |
| Linear regression | $35,606 |
| Ridge | $30,853 |
| Random forest | $30,094 |
| Ridge (log target) | $28,754 |
| Gradient boosting (log target) | $27,127 |
| Lasso (log target + log skewed features) | $26,963 |
| Ridge (log target + log skewed features) | $26,579 |
| Ridge + gradient boosting blend (equal weights) | $25,866 |
| Ridge + GBR + Huber GBR (weights 0.5 / 0.2 / 0.3) | $25,817 |
| **Weighted blend + out-of-range rule (final)** | **$25,903** |

Kaggle public leaderboard RMSE:

| Submission | Public RMSE |
|---|---|
| Blend | $45,729 |
| Blend + out-of-range rule | $17,990 |
| Weighted blend + out-of-range rule (final) | $17,884 |

The big drop shows how much one badly extrapolated house can affect RMSE on a small test set.

One training house (Id 524: very large, partial sale, sold for far less than its size suggests) accounts for a large part of the CV error. We kept it because the test set contains a house with the same profile.

## Requirements

Python 3 with pandas, numpy, scikit-learn (1.2+), matplotlib, seaborn.

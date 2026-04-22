"""
NBA Playoff Outcome Prediction - Logistic Regression (from scratch)
CS830 Final Project - Andrew Elkin

Implements binary logistic regression using NumPy only.
No ML libraries (sklearn, etc.) are used for the model itself.

Implemented from scratch:
  - Sigmoid activation
  - Binary cross-entropy loss
  - Gradient descent weight updates
  - Z-score normalization (fit on train, applied to test)
  - L2 regularization
  - Leave-one-season-out cross-validation

Library usage:
  - NumPy: array math only (dot products, exp, mean, std)
  - pandas: data loading and DataFrame manipulation
"""

import numpy as np
import pandas as pd
import os

# ── Feature configuration ─────────────────────────────────────────────────────
#
# Dean Oliver's Four Factors — applied to both offense and defense.
# All features are differentials (home team minus away team), so a positive
# value always means the home team has the advantage in that dimension.
#
# Expected coefficient signs after training:
#   DIFF_EFG_PCT      +   home shoots more efficiently  → home wins more
#   DIFF_OPP_EFG_PCT  -   home allows more eff. shooting → away wins more
#   DIFF_TOV_PCT      -   home turns ball over more/poss → away wins more
#   DIFF_OPP_TOV_PCT  +   home forces more turnovers    → home wins more
#   DIFF_ORB_PCT      +   home crashes boards more      → home wins more
#   DIFF_DRB_PCT      +   home secures def. boards more → home wins more
#   DIFF_FTR          +   home gets to line more        → home wins more
#   DIFF_OPP_FTR      -   home fouls opponent more      → away wins more
#   H2H_WIN_PCT       +   home won more h2h matchups    → home wins more

FEATURE_COLS = [
    "DIFF_EFG_PCT",      # offensive effective FG% (3s count 1.5x)
    "DIFF_OPP_EFG_PCT",  # defensive eFG% allowed
    "DIFF_TOV_PCT",      # turnover rate per possession
    "DIFF_OPP_TOV_PCT",  # forced turnover rate
    "DIFF_ORB_PCT",      # offensive rebound rate
    "DIFF_DRB_PCT",      # defensive rebound rate
    "DIFF_FTR",          # free throw attempt rate
    "DIFF_OPP_FTR",      # opponent free throw rate allowed
    "H2H_WIN_PCT",       # regular-season head-to-head record
]

LABEL_COL  = "LABEL"
SEASON_COL = "SEASON"


# ── Logistic Regression ───────────────────────────────────────────────────────

class LogisticRegression:
    """
    Binary logistic regression trained via batch gradient descent.

    Normalization is handled internally:
      - fit() computes and stores mu and sigma from the training set
      - predict_proba() applies those same stored values
    This ensures test data is never used to compute normalization params.
    """

    def __init__(self, learning_rate=0.1, epochs=1000, lambda_=0.0, verbose=False):
        self.lr      = learning_rate
        self.epochs  = epochs
        self.lambda_ = lambda_
        self.verbose = verbose

        self.w            = None
        self.b            = None
        self.mu           = None
        self.sigma        = None
        self.loss_history = []

    def _sigmoid(self, z):
        """
        Numerically stable sigmoid. Uses two equivalent forms to avoid
        overflow in exp() for large positive or negative z.
        """
        return np.where(
            z >= 0,
            1.0 / (1.0 + np.exp(-z)),
            np.exp(z) / (1.0 + np.exp(z))
        )

    def _normalize(self, X):
        return (X - self.mu) / self.sigma

    def _compute_loss(self, y_hat, y):
        """
        Binary cross-entropy + optional L2 penalty on weights (not bias).
        Clips predictions away from 0/1 to prevent log(0).
        """
        eps   = 1e-15
        y_hat = np.clip(y_hat, eps, 1 - eps)
        ce    = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        l2    = (self.lambda_ / 2.0) * np.sum(self.w ** 2)
        return ce + l2

    def fit(self, X, y):
        """
        Train via batch gradient descent.

        Forward:   z = Xw + b,  y_hat = sigmoid(z)
        Loss:      L = cross-entropy(y_hat, y) + lambda/2 * ||w||^2
        Gradients: dL/dw = (1/N) * X^T * (y_hat - y) + lambda * w
                   dL/db = mean(y_hat - y)
        Update:    w = w - lr * dL/dw
                   b = b - lr * dL/db
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        N, n_features = X.shape

        self.mu    = X.mean(axis=0)
        self.sigma = X.std(axis=0)
        self.sigma = np.where(self.sigma == 0, 1.0, self.sigma)

        X_norm = self._normalize(X)

        self.w = np.zeros(n_features)
        self.b = 0.0
        self.loss_history = []

        for epoch in range(self.epochs):
            z     = X_norm @ self.w + self.b
            y_hat = self._sigmoid(z)

            loss = self._compute_loss(y_hat, y)
            self.loss_history.append(loss)

            error  = y_hat - y
            grad_w = (1.0 / N) * (X_norm.T @ error) + self.lambda_ * self.w
            grad_b = np.mean(error)

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

            if self.verbose and (epoch + 1) % 100 == 0:
                print(f"    epoch {epoch+1:4d}/{self.epochs}  loss={loss:.5f}")

        return self

    def predict_proba(self, X):
        """P(home team wins) for each sample. Applies stored normalization."""
        X_norm = self._normalize(np.array(X, dtype=float))
        return self._sigmoid(X_norm @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        """Binary predictions: 1 = home wins, 0 = away wins."""
        return (self.predict_proba(X) >= threshold).astype(int)


# ── Baseline predictor ────────────────────────────────────────────────────────

def baseline_accuracy(df):
    """
    Simple baseline: predict home team wins iff DIFF_WIN_PCT > 0
    (always pick the team with the better regular-season record).
    This is the bar our logistic regression must beat.
    """
    preds = (df["DIFF_WIN_PCT"] > 0).astype(int)
    return float(np.mean(preds.values == df[LABEL_COL].values))


# ── Leave-one-season-out cross-validation ─────────────────────────────────────

def cross_validate(df, feature_cols=FEATURE_COLS, **model_kwargs):
    """
    Leave-one-season-out cross-validation.

    For each season S: train on all other seasons, evaluate on S.
    This respects temporal structure — no future seasons leak into training.
    """
    seasons = sorted(df[SEASON_COL].unique())
    results = []

    for test_season in seasons:
        train_df = df[df[SEASON_COL] != test_season].copy()
        test_df  = df[df[SEASON_COL] == test_season].copy()

        if train_df.empty or test_df.empty:
            continue

        X_train = train_df[feature_cols].values
        y_train = train_df[LABEL_COL].values
        X_test  = test_df[feature_cols].values
        y_test  = test_df[LABEL_COL].values

        model = LogisticRegression(**model_kwargs)
        model.fit(X_train, y_train)

        probs    = model.predict_proba(X_test)
        preds    = (probs >= 0.5).astype(int)
        accuracy = float(np.mean(preds == y_test))

        eps      = 1e-15
        probs_c  = np.clip(probs, eps, 1 - eps)
        log_loss = float(-np.mean(
            y_test * np.log(probs_c) + (1 - y_test) * np.log(1 - probs_c)
        ))

        results.append({
            "season":   test_season,
            "n_games":  int(len(y_test)),
            "accuracy": accuracy,
            "log_loss": log_loss,
        })

    return pd.DataFrame(results)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    data_path = os.path.join("data", "training_data.csv")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} examples across {df[SEASON_COL].nunique()} seasons.\n")

    base_acc = baseline_accuracy(df)
    print(f"Baseline accuracy (pick better record): {base_acc:.3f}\n")

    print("Leave-one-season-out cross-validation:")
    print("-" * 52)
    cv_results = cross_validate(
        df,
        feature_cols=FEATURE_COLS,
        learning_rate=0.1,
        epochs=1000,
        lambda_=0.01,
        verbose=False,
    )

    for _, row in cv_results.iterrows():
        print(f"  {row['season']}:  acc={row['accuracy']:.3f}  "
              f"log_loss={row['log_loss']:.3f}  (n={row['n_games']})")

    print("-" * 52)
    print(f"  Mean accuracy:  {cv_results['accuracy'].mean():.3f}  "
          f"(baseline: {base_acc:.3f})")
    print(f"  Mean log-loss:  {cv_results['log_loss'].mean():.3f}\n")

    print("Training final model on full dataset...")
    X_all = df[FEATURE_COLS].values
    y_all = df[LABEL_COL].values

    final_model = LogisticRegression(
        learning_rate=0.1, epochs=1000, lambda_=0.01, verbose=True
    )
    final_model.fit(X_all, y_all)

    final_preds = (final_model.predict_proba(X_all) >= 0.5).astype(int)
    print(f"\nFinal model train accuracy: {np.mean(final_preds == y_all):.3f}")

    print("\nLearned feature weights (normalized scale):")
    print(f"  {'feature':<24} weight")
    print(f"  {'-'*36}")
    for feat, w in sorted(zip(FEATURE_COLS, final_model.w), key=lambda x: -abs(x[1])):
        bar  = "█" * int(abs(w) * 20)
        sign = "+" if w >= 0 else "-"
        print(f"  {feat:<24} {sign}{abs(w):.4f}  {bar}")
    print(f"  {'bias':<24} {final_model.b:+.4f}")


if __name__ == "__main__":
    main()
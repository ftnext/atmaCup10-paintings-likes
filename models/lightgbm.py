import numpy as np
from sklearn.metrics import mean_squared_error

import lightgbm as lgbm

# ref: https://www.guruguru.science/competitions/16/discussions/185c7dc6-5e3a-49c6-9c30-41bf007cc694/  # NOQA
params = {
    # 目的関数. これの意味で最小となるようなパラメータを探します.
    "objective": "rmse",
    # 学習率. 小さいほどなめらかな決定境界が作られて性能向上に繋がる場合が多いです、
    # がそれだけ木を作るため学習に時間がかかります
    "learning_rate": 0.1,
    # L2 Regularization
    "reg_lambda": 1.0,
    # こちらは L1
    "reg_alpha": 0.1,
    # 木の深さ. 深い木を許容するほどより複雑な交互作用を考慮するようになります
    "max_depth": 5,
    # 木の最大数. early_stopping という枠組みで木の数は制御されるようにしていますのでとても大きい値を指定しておきます.
    "n_estimators": 10000,
    # 木を作る際に考慮する特徴量の割合. 1以下を指定すると特徴をランダムに欠落させます。
    # 小さくすることで, まんべんなく特徴を使うという効果があります.
    "colsample_bytree": 0.5,
    # 最小分割でのデータ数. 小さいとより細かい粒度の分割方法を許容します.
    "min_child_samples": 10,
    # bagging の頻度と割合
    "subsample_freq": 3,
    "subsample": 0.9,
    # 特徴重要度計算のロジック
    "importance_type": "gain",
    "random_state": 71,
}


def fit_lgbm(X, y, cv, params=None, verbose=50):
    if params is None:
        params = {}

    models = []
    oof_pred = np.zeros_like(y, dtype=float)

    for i, (idx_train, idx_valid) in enumerate(cv):
        x_train, y_train = X[idx_train], y[idx_train]
        x_valid, y_valid = X[idx_valid], y[idx_valid]

        clf = lgbm.LGBMRegressor(**params)
        clf.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            early_stopping_rounds=100,
            verbose=verbose,
        )

        pred_i = clf.predict(x_valid)
        oof_pred[idx_valid] = pred_i
        models.append(clf)
        print(
            f"Fold {i} RMSLE: {mean_squared_error(y_valid, pred_i) ** .5:.4f}"
        )

    score = mean_squared_error(y, oof_pred) ** 0.5
    print("-" * 50)
    print(f"FINISHED | Whole RMSLE: {score:.4f}")
    return oof_pred, models

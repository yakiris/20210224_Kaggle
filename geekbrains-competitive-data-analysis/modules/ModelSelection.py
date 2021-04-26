import numpy as np
import pandas as pd
import time

from sklearn.model_selection import train_test_split, KFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
import optuna

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

class ModelSelection():

    def __init__(self, X: pd.DataFrame,
                       y: pd.Series,
                       stata_model: pd.DataFrame = None,
                       guess: str = None):
        
        self.X = X.copy()
        self.y = y.copy()
        self.stata_model = stata_model
        self.guess = guess
        self.start = time.perf_counter()

    @staticmethod
    def data_separation(data):
        '''Функция разбиения общего датасета на итоговые: обучающую и тестовую подвыборки.'''
        data = data.reset_index(drop=True)
        data = data.replace(np.inf, np.nan)
        data = data.replace(-np.inf, np.nan)

        mask = data['TARGET'].isnull()
        features_drop = ['APPLICATION_NUMBER', 'TARGET']

        data_train = data.loc[~mask].reset_index(drop=True)
        data_valid = data.loc[mask].reset_index(drop=True)

        X, X_final = data_train, data_valid
        y, final_id = X['TARGET'], X_final['APPLICATION_NUMBER']

        X = X.drop(features_drop, axis=1)
        X_final = X_final.drop(features_drop, axis=1)

        return data_train, X, y, X_final, final_id

    def get_train_valid_test(self):
        '''Функция разбиения датасета на обучающие, валидационные и тестовые подвыборки'''

        X_train, X_test = train_test_split(self.X, train_size=0.7, shuffle=True, random_state=15)
        y_train, y_test = train_test_split(self.y, train_size=0.7, shuffle=True, random_state=15)

        X_train, X_valid = train_test_split(X_train, train_size=0.7, shuffle=True, random_state=15)
        y_train, y_valid = train_test_split(y_train, train_size=0.7, shuffle=True, random_state=15)

        return X_train, y_train, X_valid, y_valid, X_test, y_test
        
    def cross_validation(self, model):
        '''Функция кросс-валидации KFold на 5 фолдах и hold-out на 3 подвыборках'''

        cv_strategy = KFold(n_splits=5, random_state=1, shuffle=True)

        fold_train_scores, fold_valid_scores = [], []
        oof_preds = np.zeros(self.X.shape[0])

        for train_idx, valid_idx in cv_strategy.split(self.X, self.y):
            X_train_cv, X_valid_cv = self.X.loc[train_idx], self.X.loc[valid_idx]
            y_train_cv, y_valid_cv = self.y.loc[train_idx], self.y.loc[valid_idx]

            model.fit(X_train_cv, y_train_cv)

            oof_preds[valid_idx] = model.predict_proba(X_valid_cv)[:, 1]
            y_train_cv_pred = model.predict_proba(X_train_cv)[:, 1]
            y_valid_cv_pred = model.predict_proba(X_valid_cv)[:, 1]
            fold_train_scores.append(roc_auc_score(y_train_cv, y_train_cv_pred))
            fold_valid_scores.append(roc_auc_score(y_valid_cv, y_valid_cv_pred))

        train_scores = round(np.mean(fold_train_scores), 4)
        valid_scores = round(np.mean(fold_valid_scores), 4)

        conf_interval = 0.95 

        left_bound = np.percentile(fold_valid_scores, ((1 - conf_interval) / 2) * 100)
        right_bound = np.percentile(fold_valid_scores, (conf_interval + ((1 - conf_interval) / 2)) * 100)
        interval = f'{round(left_bound, 3)}/{round(right_bound, 3)}'

        return oof_preds, train_scores, valid_scores, interval
    
    def collection_statistic(self, model):
        '''Функция логирования статистических показателей.'''
        oof_preds, train_scores, valid_scores, interval = self.cross_validation(model)

        X_train, y_train, X_valid, y_valid, X_test, y_test = self.get_train_valid_test()

        self.stata_model.loc[f'{self.guess}', 'train_Hold_Out'] = \
                round(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]), 4)
        self.stata_model.loc[f'{self.guess}', 'valid_Hold_Out'] = \
                round(roc_auc_score(y_valid, model.predict_proba(X_valid)[:, 1]), 4)
        self.stata_model.loc[f'{self.guess}', 'test'] = \
                round(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]), 4)
        self.stata_model.loc[f'{self.guess}', 'train_KFold'] = train_scores
        self.stata_model.loc[f'{self.guess}', 'valid_KFold'] = valid_scores
        self.stata_model.loc[f'{self.guess}', 'valid_interval'] = interval

        stop = time.perf_counter()
        self.stata_model.loc[f'{self.guess}', 'time_model'] = f'{stop - self.start:0.1f}'
        
        return oof_preds, self.stata_model
    
    def log_regression(self, param_lr):
        '''LogisticRegression'''

        self.X.fillna(0, inplace=True)

        X_train, y_train, _, _, _, _ = self.get_train_valid_test()

        model_lr = LogisticRegression(**param_lr)
        model_lr.fit(X_train, y_train)
        
        _, self.stata_model = self.collection_statistic(model_lr)
        
        return model_lr, self.stata_model
    
    def dt_classifier(self, param_dt):
        '''DecisionTreeClassifier'''

        self.X.fillna(0, inplace=True)

        X_train, y_train, _, _, _, _ = self.get_train_valid_test()

        model_dt = DecisionTreeClassifier(**param_dt)
        model_dt.fit(X_train, y_train)
        
        _, self.stata_model = self.collection_statistic(model_dt)
        
        return model_dt, self.stata_model
    
    def rf_classifier(self, param_rf):
        '''RandomForestClassifier'''

        self.X.fillna(0, inplace=True)

        X_train, y_train, _, _, _, _ = self.get_train_valid_test()

        model_rf = RandomForestClassifier(**param_rf)
        model_rf.fit(X_train, y_train)

        _, self.stata_model = self.collection_statistic(model_rf)
        
        return model_rf, self.stata_model
        
    def xgb_classifier(self, params_xgb, feature_importances=0):
        '''XGBClassifier'''

        X_train, y_train, X_valid, y_valid, _, _ = self.get_train_valid_test()

        model_xgb = XGBClassifier(booster="gbtree",
                                objective="binary:logistic",
                                eval_metric="auc",
                                random_state=42,
                                nthread=6,
                                **params_xgb)

        model_xgb.fit(X=X_train, y=y_train,
                    eval_set=[(X_train, y_train), (X_valid, y_valid)],
                    early_stopping_rounds=50,
                    eval_metric="auc",
                    verbose=100)

        if feature_importances == 0:
            oof_preds, self.stata_model = self.collection_statistic(model_xgb)

            return model_xgb, oof_preds, self.stata_model
        else:
            drop_features = self.get_permutation_importance(model_xgb, X_valid, y_valid)

            return drop_features

    def lgbm_classifier(self, params_lgbm, feature_importances=0):
        '''LGBMClassifier'''

        X_train, y_train, X_valid, y_valid, _, _ = self.get_train_valid_test()

        model_lgbm = LGBMClassifier(objective="binary",
                                    boosting_type="gbdt",
                                    n_jobs=6,
                                    random_state=42,
                                    **params_lgbm)

        model_lgbm.fit(X=X_train, y=y_train,
                    eval_set=[(X_train, y_train), (X_valid, y_valid)],
                    early_stopping_rounds=50,
                    eval_metric="auc",
                    verbose=100)

        if feature_importances == 0:
            oof_preds, self.stata_model = self.collection_statistic(model_lgbm)

            return model_lgbm, oof_preds, self.stata_model
        else:
            drop_features = self.get_permutation_importance(model_lgbm, X_valid, y_valid)

            return drop_features

    def cb_classifier(self, params_cb, feature_importances=0):
        '''CatBoostClassifier'''

        X_train, y_train, X_valid, y_valid, _, _ = self.get_train_valid_test()

        model_cb = CatBoostClassifier(loss_function="Logloss",
                                        eval_metric="AUC",
                                        task_type="CPU",
                                        verbose=100,
                                        random_state=42,
                                        # max_bin=20,
                                        # l2_leaf_reg=10,
                                        # thread_count=6,
                                        **params_cb)
        model_cb.fit(X_train, y_train,
                         eval_set=[(X_train, y_train), (X_valid, y_valid)],
                         early_stopping_rounds=50)

        if feature_importances == 0:
            oof_preds, self.stata_model = self.collection_statistic(model_cb)

            return model_cb, oof_preds, self.stata_model
        else:
            drop_features = self.get_permutation_importance(model_cb, X_valid, y_valid)

            return drop_features

    def get_permutation_importance(self, model, X_valid, y_valid):
        '''Функция отбора важных признаков.
           Механизм feature_importances'''

        importance = permutation_importance(model, X_valid, y_valid, scoring="roc_auc", random_state=27)
        importance_scores = pd.DataFrame({"features": X_valid.columns,
                                          "importance-mean": importance.importances_mean})
        decrease_scores = importance_scores[importance_scores["importance-mean"] <= 0]

        return decrease_scores['features']

    def xgb_optuna(self, trial):
        '''XGBClassifier для подбора гиперпараметров'''
        X_train, y_train, X_valid, y_valid, _, _ = self.get_train_valid_test()

        xgb_tuna = XGBClassifier(
            booster="gbtree",
            objective="binary:logistic",
            eval_metric="auc",
            random_state=42,
            nthread=6,

            n_estimators=trial.suggest_int('n_estimators', 500, 1000),
            max_depth=trial.suggest_int('max_depth', 3, 10),
            learning_rate=trial.suggest_loguniform("learning_rate", 0.01, 0.1),
            min_child_weight=trial.suggest_int('min_child_weight', 20, 100),
            subsample=trial.suggest_float('subsample', 0, 1),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0, 1),
            colsample_bylevel=trial.suggest_float('colsample_bylevel', 0, 1),
            gamma=trial.suggest_int('gamma', 0, 20),
            eta=trial.suggest_float('eta', 0, 1),
            reg_lambda=trial.suggest_float('reg_lambda', 0, 1),
            reg_alpha=trial.suggest_float('reg_alpha', 0, 1),
            )

        xgb_tuna.fit(X=X_train, y=y_train,
                      eval_set=[(X_train, y_train), (X_valid, y_valid)],
                      early_stopping_rounds=50,
                      eval_metric="auc",
                      verbose=100)

        xgb_predict_test = xgb_tuna.predict_proba(X_valid)[:, 1]
        score = roc_auc_score(y_valid, xgb_predict_test)

        return score

    def lgbm_optuna(self, trial):
        '''LGBMClassifier для подбора гиперпараметров
           Механизм tuning_parametrs'''

        X_train, y_train, X_valid, y_valid, _, _ = self.get_train_valid_test()

        lgbm_tuna = LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            n_jobs=6,
            random_state=42,

            n_estimators=trial.suggest_int('n_estimators', 500, 1000),
            max_depth=trial.suggest_int('max_depth', 3, 10),
            learning_rate=trial.suggest_loguniform("learning_rate", 0.01, 0.1),
            num_leaves=trial.suggest_int('num_leaves', 10, 70),
            min_child_weight=trial.suggest_int('min_child_weight', 20, 100),
            reg_lambda=trial.suggest_float('reg_lambda', 0, 1),
            reg_alpha=trial.suggest_float('reg_alpha', 0, 1),
        )

        lgbm_tuna.fit(X=X_train, y=y_train,
                          eval_set=[(X_train, y_train), (X_valid, y_valid)],
                          early_stopping_rounds=50,
                          eval_metric="auc",
                          verbose=-1)

        lgbm_predict_test = lgbm_tuna.predict_proba(X_valid)[:, 1]
        score = roc_auc_score(y_valid, lgbm_predict_test)

        return score

    def cb_optuna(self, trial):
        '''CatBoostClassifier для подбора гиперпараметров
           Механизм tuning_parametrs'''

        X_train, y_train, X_valid, y_valid, _, _ = self.get_train_valid_test()

        cb_tuna = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            task_type="CPU",
            verbose=100,
            random_state=42,

            n_estimators=trial.suggest_int('n_estimators', 500, 1000),
            max_depth=trial.suggest_int('max_depth', 3, 10),
            learning_rate=trial.suggest_loguniform("learning_rate", 0.01, 0.1),
            max_bin=trial.suggest_int('max_bin', 0, 100),
            l2_leaf_reg=trial.suggest_int('l2_leaf_reg', 0, 10),
            thread_count=trial.suggest_int('thread_count', 0, 10),
            )

        cb_tuna.fit(X_train, y_train,
                     eval_set=[(X_train, y_train), (X_valid, y_valid)],
                     early_stopping_rounds=50)

        cb_predict_test = cb_tuna.predict_proba(X_valid)[:, 1]
        score = roc_auc_score(y_valid, cb_predict_test)

        return score

    def select_parametrs(self, name_model):
        '''Функция подбора гиперпараметров.
           Механизм tuning_parametrs'''

        if name_model == 'xgb':
            model = self.xgb_optuna
        elif name_model == 'lgbm':
            model = self.lgbm_optuna
        elif name_model == 'cb':
            model = self.cb_optuna

        lgbm_study = optuna.create_study(direction="maximize")
        lgbm_study.optimize(model, n_trials=20)
        trial = lgbm_study.best_trial

        print('*'*100)
        print(f"Number of finished trials {name_model}: {len(lgbm_study.trials)}")
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"'{key}': {value},")
        print('*' * 100)

    @staticmethod
    def drop_outliers(data, model, loss=0.8):
        '''Функция удаления строк с выбросами из обучающей подвыборки.
           Механизм drop_outliers'''

        X = data.drop(['APPLICATION_NUMBER', 'TARGET'], axis=1)
        df_pred = data[['APPLICATION_NUMBER', 'TARGET']]
        df_pred['pred'] = model.predict_proba(X)[:, 1]
        df_pred['loss'] = np.abs(df_pred['TARGET'] - df_pred['pred'])
        drop_list = df_pred[df_pred['loss'] > loss].index.to_list()
        data = data.drop(drop_list, axis=0)

        return data




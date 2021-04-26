import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier


class FeatureEngineering():

    def __init__(self, X: pd.DataFrame):
        self.X = X.copy()
        self.cat_features = X.select_dtypes(include=np.object).columns.to_list()

    def label_encoder(self):

        le = LabelEncoder()
        for feature in self.cat_features:
            cat_value = list(self.X[feature].values.astype('str'))
            le.fit(cat_value)
            self.X[feature] = le.transform(cat_value)

        return self.X

    def one_hot_encoder(self):

        self.X = pd.get_dummies(self.X, columns=self.cat_features)

        return self.X

    def frequency_encoder(self):

        for feature in self.cat_features:
            freq_value = self.X[feature].value_counts(normalize=True)
            self.X[feature] = self.X[feature].map(freq_value)

        return self.X

    def cp_preprocessing(self):

        self.X.loc[self.X['GENDER'] == 'XNA', 'GENDER'] = np.nan
        self.X.loc[self.X['DAYS_ON_LAST_JOB'] == 365243, 'DAYS_ON_LAST_JOB'] = np.nan
        self.X['EDUCATION_LEVEL'] = self.X['EDUCATION_LEVEL'].map({'Lower secondary': 0,
                                                     'Secondary / secondary special': 1,
                                                                 'Incomplete higher': 2,
                                                                  'Higher education': 3,
                                                                   'Academic degree': 4})

        self.X['OWN_CAR_AGE'] = self.X['OWN_CAR_AGE'].fillna(50)
        self.X.drop('CHILDRENS', axis=1, inplace=True)
        self.X.drop('FAMILY_SIZE', axis=1, inplace=True)
        self.X.drop('FLAG_PHONE', axis=1, inplace=True)
        self.X.drop('FLAG_EMAIL', axis=1, inplace=True)
        self.X.drop('AMT_REQ_CREDIT_BUREAU_HOUR', axis=1, inplace=True)
        self.X.drop('AMT_REQ_CREDIT_BUREAU_DAY', axis=1, inplace=True)
        self.X.drop('AMT_REQ_CREDIT_BUREAU_WEEK', axis=1, inplace=True)
        self.X.drop('AMT_REQ_CREDIT_BUREAU_MON', axis=1, inplace=True)

        return self.X

    def get_probability_target(self, data, name_feature):

        X = data.loc[data['TARGET'].notnull()].reset_index(drop=True)
        y = X['TARGET']
        final_id = data['APPLICATION_NUMBER']

        X = X.drop(['APPLICATION_NUMBER', 'TARGET'], axis=1)
        X_final = data.drop(['APPLICATION_NUMBER', 'TARGET'], axis=1)

        model = LGBMClassifier(max_depth=2, random_state=1)
        model.fit(X, y)

        y_prob = model.predict_proba(X_final)[:, 1]
        df_prob = pd.DataFrame({'APPLICATION_NUMBER': final_id, name_feature: y_prob})

        return df_prob

    def cp_prob_rating(self):

        spam_df = self.X[['APPLICATION_NUMBER', 'EXTERNAL_SCORING_RATING_1', 'TARGET']]
        df_prob = self.get_probability_target(spam_df, 'PROB_RATING_1')
        self.X = self.X.merge(df_prob, how='left', on='APPLICATION_NUMBER')

        spam_df = self.X[['APPLICATION_NUMBER', 'EXTERNAL_SCORING_RATING_2', 'TARGET']]
        df_prob = self.get_probability_target(spam_df, 'PROB_RATING_2')
        self.X = self.X.merge(df_prob, how='left', on='APPLICATION_NUMBER')

        spam_df = self.X[['APPLICATION_NUMBER', 'EXTERNAL_SCORING_RATING_3', 'TARGET']]
        df_prob = self.get_probability_target(spam_df, 'PROB_RATING_3')
        self.X = self.X.merge(df_prob, how='left', on='APPLICATION_NUMBER')

        return self.X

    def cp_age_mean_salary(self):

        cp_temp = self.X[['AGE', 'TOTAL_SALARY']].groupby(['AGE'], as_index=False).mean()
        cp_temp.rename(columns={'TOTAL_SALARY': 'AGE_MEAN_SALARY'}, inplace=True)
        self.X = self.X.merge(cp_temp, how='left', on='AGE')

        return self.X

    def cp_age_mean_annuity(self):

        cp_temp = self.X[['AGE', 'AMOUNT_ANNUITY']].groupby(['AGE'], as_index=False).mean()
        cp_temp.rename(columns={'AMOUNT_ANNUITY': 'AGE_MEAN_ANNUITY'}, inplace=True)
        self.X = self.X.merge(cp_temp, how='left', on='AGE')

        return self.X

    def cp_age_mean_credit(self):

        cp_temp = self.X[['AGE', 'AMOUNT_CREDIT']].groupby(['AGE'], as_index=False).mean()
        cp_temp.rename(columns={'AMOUNT_CREDIT': 'AGE_MEAN_CREDIT'}, inplace=True)
        self.X = self.X.merge(cp_temp, how='left', on='AGE')

        return self.X

    def cp_salary_ratio_age(self):

        self.X['SALARY_RATIO_AGE'] = self.X['TOTAL_SALARY'] / self.X['AGE']

        return self.X

    def cp_annuity_ratio_age(self):

        self.X['ANNUITY_RATIO_AGE'] = self.X['AMOUNT_ANNUITY'] / self.X['AGE']

        return self.X

    def cp_credit_ratio_age(self):

        self.X['CREDIT_RATIO_AGE'] = self.X['AMOUNT_CREDIT'] / self.X['AGE']

        return self.X

    def cp_age_mean_rating1(self):

        cp_temp = self.X[['AGE', 'EXTERNAL_SCORING_RATING_1']].groupby(['AGE'], as_index=False).mean()
        cp_temp.rename(columns={'EXTERNAL_SCORING_RATING_1': 'AGE_MEAN_RATING_1'}, inplace=True)
        self.X = self.X.merge(cp_temp, how='left', on='AGE')

        return self.X

    def cp_age_mean_rating2(self):

        cp_temp = self.X[['AGE', 'EXTERNAL_SCORING_RATING_2']].groupby(['AGE'], as_index=False).mean()
        cp_temp.rename(columns={'EXTERNAL_SCORING_RATING_2': 'AGE_MEAN_RATING_2'}, inplace=True)
        self.X = self.X.merge(cp_temp, how='left', on='AGE')

        return self.X

    def cp_age_mean_rating3(self):

        cp_temp = self.X[['AGE', 'EXTERNAL_SCORING_RATING_3']].groupby(['AGE'], as_index=False).mean()
        cp_temp.rename(columns={'EXTERNAL_SCORING_RATING_3': 'AGE_MEAN_RATING_3'}, inplace=True)
        self.X = self.X.merge(cp_temp, how='left', on='AGE')

        return self.X

    def cp_rating1_ratio_age(self):

        self.X['RATING_1_RATIO_AGE'] = self.X['EXTERNAL_SCORING_RATING_1'] / self.X['AGE']

        return self.X

    def cp_prob_rating1_ratio_age(self):

        self.X['PROB_RATING_1_RATIO_AGE'] = self.X['PROB_RATING_1'] / self.X['AGE']

        return self.X

    def cp_rating2_ratio_age(self):

        self.X['RATING_2_RATIO_AGE'] = self.X['EXTERNAL_SCORING_RATING_2'] / self.X['AGE']

        return self.X

    def cp_prob_rating2_ratio_age(self):

        self.X['PROB_RATING_2_RATIO_AGE'] = self.X['PROB_RATING_2'] / self.X['AGE']

        return self.X

    def cp_rating3_ratio_age(self):

        self.X['RATING_3_RATIO_AGE'] = self.X['EXTERNAL_SCORING_RATING_3'] / self.X['AGE']

        return self.X

    def cp_prob_rating3_ratio_age(self):

        self.X['PROB_RATING_3_RATIO_AGE'] = self.X['PROB_RATING_3'] / self.X['AGE']

        return self.X

    def cp_salary_ratio_job(self):

        self.X['SALARY_RATIO_JOB'] = self.X['TOTAL_SALARY'] / self.X['DAYS_ON_LAST_JOB']

        return self.X

    def cp_annuity_ratio_job(self):

        self.X['ANNUITY_RATIO_JOB'] = self.X['AMOUNT_ANNUITY'] / self.X['DAYS_ON_LAST_JOB']

        return self.X

    def cp_credit_ratio_job(self):

        self.X['CREDIT_RATIO_JOB'] = self.X['AMOUNT_CREDIT'] / self.X['DAYS_ON_LAST_JOB']

        return self.X

    def cp_rating1_ratio_job(self):

        self.X['RATING_1_RATIO_JOB'] = self.X['EXTERNAL_SCORING_RATING_1'] / self.X['DAYS_ON_LAST_JOB']

        return self.X

    def cp_prob_rating1_ratio_job(self):

        self.X['PROB_RATING_1_RATIO_JOB'] = self.X['PROB_RATING_1'] / self.X['DAYS_ON_LAST_JOB']

        return self.X

    def cp_rating2_ratio_job(self):

        self.X['RATING_2_RATIO_JOB'] = self.X['EXTERNAL_SCORING_RATING_2'] / self.X['DAYS_ON_LAST_JOB']

        return self.X

    def cp_prob_rating2_ratio_job(self):

        self.X['PROB_RATING_2_RATIO_JOB'] = self.X['PROB_RATING_2'] / self.X['DAYS_ON_LAST_JOB']

        return self.X

    def cp_rating3_ratio_job(self):

        self.X['RATING_3_RATIO_JOB'] = self.X['EXTERNAL_SCORING_RATING_3'] / self.X['DAYS_ON_LAST_JOB']

        return self.X

    def cp_prob_rating3_ratio_job(self):

        self.X['PROB_RATING_3_RATIO_JOB'] = self.X['PROB_RATING_3'] / self.X['DAYS_ON_LAST_JOB']

        return self.X

    def cp_salary_ratio_car(self):

        self.X['SALARY_RATIO_CAR'] = self.X['TOTAL_SALARY'] / self.X['OWN_CAR_AGE']

        return self.X

    def cp_annuity_ratio_car(self):

        self.X['ANNUITY_RATIO_CAR'] = self.X['AMOUNT_ANNUITY'] / self.X['OWN_CAR_AGE']

        return self.X

    def cp_credit_ratio_car(self):

        self.X['CREDIT_RATIO_CAR'] = self.X['AMOUNT_CREDIT'] / self.X['OWN_CAR_AGE']

        return self.X

    def cp_rating1_ratio_car(self):

        self.X['RATING_1_RATIO_CAR'] = self.X['EXTERNAL_SCORING_RATING_1'] / self.X['OWN_CAR_AGE']

        return self.X

    def cp_rating2_ratio_car(self):

        self.X['RATING_2_RATIO_CAR'] = self.X['EXTERNAL_SCORING_RATING_2'] / self.X['OWN_CAR_AGE']

        return self.X

    def cp_rating3_ratio_car(self):

        self.X['RATING_3_RATIO_CAR'] = self.X['EXTERNAL_SCORING_RATING_3'] / self.X['OWN_CAR_AGE']

        return self.X

    def cp_edu_mean_salary(self):

        cp_temp = self.X[['EDUCATION_LEVEL', 'TOTAL_SALARY']].groupby(['EDUCATION_LEVEL'], as_index=False).mean()
        cp_temp.rename(columns={'TOTAL_SALARY': 'EDU_MEAN_SALARY'}, inplace=True)
        self.X = self.X.merge(cp_temp, how='left', on='EDUCATION_LEVEL')

        return self.X

    def cp_edu_mean_annuity(self):

        cp_temp = self.X[['EDUCATION_LEVEL', 'AMOUNT_ANNUITY']].groupby(['EDUCATION_LEVEL'], as_index=False).mean()
        cp_temp.rename(columns={'AMOUNT_ANNUITY': 'EDU_MEAN_ANNUITY'}, inplace=True)
        self.X = self.X.merge(cp_temp, how='left', on='EDUCATION_LEVEL')

        return self.X

    def cp_edu_mean_credit(self):

        cp_temp = self.X[['EDUCATION_LEVEL', 'AMOUNT_CREDIT']].groupby(['EDUCATION_LEVEL'], as_index=False).mean()
        cp_temp.rename(columns={'AMOUNT_CREDIT': 'EDU_MEAN_CREDIT'}, inplace=True)
        self.X = self.X.merge(cp_temp, how='left', on='EDUCATION_LEVEL')

        return self.X

    def cp_edu_mean_rating1(self):

        cp_temp = self.X[['EDUCATION_LEVEL', 'EXTERNAL_SCORING_RATING_1']].groupby(['EDUCATION_LEVEL'],
                                                                                   as_index=False).mean()
        cp_temp.rename(columns={'EXTERNAL_SCORING_RATING_1': 'EDU_MEAN_RATING_1'}, inplace=True)
        self.X = self.X.merge(cp_temp, how='left', on='EDUCATION_LEVEL')

        return self.X

    def cp_edu_mean_rating2(self):

        cp_temp = self.X[['EDUCATION_LEVEL', 'EXTERNAL_SCORING_RATING_2']].groupby(['EDUCATION_LEVEL'],
                                                                                   as_index=False).mean()
        cp_temp.rename(columns={'EXTERNAL_SCORING_RATING_2': 'EDU_MEAN_RATING_2'}, inplace=True)
        self.X = self.X.merge(cp_temp, how='left', on='EDUCATION_LEVEL')

        return self.X

    def cp_edu_mean_rating3(self):

        cp_temp = self.X[['EDUCATION_LEVEL', 'EXTERNAL_SCORING_RATING_3']].groupby(['EDUCATION_LEVEL'],
                                                                                   as_index=False).mean()
        cp_temp.rename(columns={'EXTERNAL_SCORING_RATING_3': 'EDU_MEAN_RATING_3'}, inplace=True)
        self.X = self.X.merge(cp_temp, how='left', on='EDUCATION_LEVEL')

        return self.X

    def cp_salary_ratio_region(self):

        self.X['SALARY_RATIO_REGION'] = self.X['TOTAL_SALARY'] / self.X['REGION_POPULATION']

        return self.X

    def cp_annuity_ratio_region(self):

        self.X['ANNUITY_RATIO_REGION'] = self.X['AMOUNT_ANNUITY'] / self.X['REGION_POPULATION']

        return self.X

    def cp_credit_ratio_region(self):

        self.X['CREDIT_RATIO_REGION'] = self.X['AMOUNT_CREDIT'] / self.X['REGION_POPULATION']

        return self.X

    def cp_rating1_ratio_region(self):

        self.X['RATING_1_RATIO_REGION'] = self.X['EXTERNAL_SCORING_RATING_1'] / self.X['REGION_POPULATION']

        return self.X

    def cp_rating2_ratio_region(self):

        self.X['RATING_2_RATIO_REGION'] = self.X['EXTERNAL_SCORING_RATING_2'] / self.X['REGION_POPULATION']

        return self.X

    def cp_rating3_ratio_region(self):

        self.X['RATING_3_RATIO_REGION'] = self.X['EXTERNAL_SCORING_RATING_3'] / self.X['REGION_POPULATION']

        return self.X

    def cp_rating_min(self):

        self.X['RATING_MIN'] = self.X[['EXTERNAL_SCORING_RATING_1', 'EXTERNAL_SCORING_RATING_2', \
                                       'EXTERNAL_SCORING_RATING_3']].min(axis=1)

        return self.X

    def cp_prob_rating_min(self):

        self.X['PROB_RATING_MIN'] = self.X[['PROB_RATING_1', 'PROB_RATING_2', 'PROB_RATING_3']].min(axis=1)

        return self.X

    def cp_rating_max(self):

        self.X['RATING_MAX'] = self.X[['EXTERNAL_SCORING_RATING_1', 'EXTERNAL_SCORING_RATING_2', \
                                       'EXTERNAL_SCORING_RATING_3']].max(axis=1)

        return self.X

    def cp_prob_rating_max(self):

        self.X['PROB_RATING_MAX'] = self.X[['PROB_RATING_1', 'PROB_RATING_2', 'PROB_RATING_3']].max(axis=1)

        return self.X

    def cp_rating_mean(self):

        self.X['RATING_MEAN'] = self.X[['EXTERNAL_SCORING_RATING_1', 'EXTERNAL_SCORING_RATING_2', \
                                        'EXTERNAL_SCORING_RATING_3']].mean(axis=1)

        return self.X

    def cp_prob_rating_mean(self):

        self.X['PROB_RATING_MEAN'] = self.X[['PROB_RATING_1', 'PROB_RATING_2', 'PROB_RATING_3']].mean(axis=1)

        return self.X

    def cp_rating_prod(self):

        self.X['RATING_PROD'] = self.X['EXTERNAL_SCORING_RATING_1'] * self.X['EXTERNAL_SCORING_RATING_2'] * \
                                self.X['EXTERNAL_SCORING_RATING_3']

        return self.X

    def cp_prob_rating_prod(self):

        self.X['PROB_RATING_PROD'] = self.X['PROB_RATING_1'] * self.X['PROB_RATING_2'] * self.X['PROB_RATING_3']

        return self.X

    def cp_rating_balance(self):

        self.X['RATING_BALANCE'] = self.X['EXTERNAL_SCORING_RATING_1'] * 2 + self.X['EXTERNAL_SCORING_RATING_2'] + \
                                   self.X['EXTERNAL_SCORING_RATING_3'] * 3

        return self.X

    def cp_prob_rating_balance(self):

        self.X['PROB_RATING_BALANCE'] = self.X['PROB_RATING_1']*2 + self.X['PROB_RATING_2'] * self.X['PROB_RATING_3']*3

        return self.X

    def cp_rating1_prod_credit(self):

        self.X['RATING1_PROD_CREDIT'] = self.X['EXTERNAL_SCORING_RATING_1'] * self.X['AMOUNT_CREDIT']

        return self.X

    def cp_rating2_prod_credit(self):

        self.X['RATING2_PROD_CREDIT'] = self.X['EXTERNAL_SCORING_RATING_2'] * self.X['AMOUNT_CREDIT']

        return self.X

    def cp_rating3_prod_credit(self):

        self.X['RATING3_PROD_CREDIT'] = self.X['EXTERNAL_SCORING_RATING_3'] * self.X['AMOUNT_CREDIT']

        return self.X

    def cp_rating1_prod_annuity(self):

        self.X['RATING1_PROD_ANNUITY'] = self.X['EXTERNAL_SCORING_RATING_1'] * self.X['AMOUNT_ANNUITY']

        return self.X

    def cp_rating2_prod_annuity(self):

        self.X['RATING2_PROD_ANNUITY'] = self.X['EXTERNAL_SCORING_RATING_2'] * self.X['AMOUNT_ANNUITY']

        return self.X

    def cp_rating3_prod_annuity(self):

        self.X['RATING3_PROD_ANNUITY'] = self.X['EXTERNAL_SCORING_RATING_3'] * self.X['AMOUNT_ANNUITY']

        return self.X

    def cp_credit_ratio_annuity(self):

        self.X['CREDIT_RATIO_ANNUITY'] = self.X['AMOUNT_CREDIT'] / self.X['AMOUNT_ANNUITY']

        return self.X

    def cp_credit_ratio_salary(self):

        self.X['CREDIT_RATIO_SALARY'] = self.X['AMOUNT_CREDIT'] / self.X['TOTAL_SALARY']

        return self.X

    def cp_salary_diff_annuity(self):

        self.X['SALARY_DIFF_ANNUITY'] = self.X['TOTAL_SALARY'] - self.X['AMOUNT_ANNUITY']

        return self.X

    def cp_age_diff_car(self):

        self.X['AGE_DIFF_CAR'] = self.X['AGE'] - self.X['OWN_CAR_AGE'] * 365

        return self.X

    def cp_age_diff_job(self):

        self.X['AGE_DIFF_JOB'] = self.X['AGE'] - self.X['DAYS_ON_LAST_JOB']

        return self.X

    def cp_job_ratio_car(self):

        self.X['JOB_RATIO_CAR'] = self.X['DAYS_ON_LAST_JOB'] / self.X['OWN_CAR_AGE'] * 365

        return self.X

    def cp_gender_min_salary(self):

        cp_temp = self.X[['GENDER', 'TOTAL_SALARY']].groupby(['GENDER'], as_index=False).min()
        cp_temp.rename(columns={'TOTAL_SALARY': 'GENDER_MIN_SALARY'}, inplace=True)
        self.X = self.X.merge(cp_temp, how='left', on='GENDER')

        return self.X

    def cp_gender_max_salary(self):

        cp_temp = self.X[['GENDER', 'TOTAL_SALARY']].groupby(['GENDER'], as_index=False).max()
        cp_temp.rename(columns={'TOTAL_SALARY': 'GENDER_MIN_SALARY'}, inplace=True)
        self.X = self.X.merge(cp_temp, how='left', on='GENDER')

        return self.X

    def cp_age_prod_credit_term(self):

        self.X['AGE_PROD_CREDIT_TERM'] = self.X['AGE'] + self.X['CREDIT_RATIO_ANNUITY'] * 30

        return self.X

    # ***************

    def ap_days_nan(self):

        self.X.loc[self.X['DAYS_DECISION'] == 365243, 'DAYS_DECISION'] = np.nan
        self.X.loc[self.X['DAYS_FIRST_DRAWING'] == 365243, 'DAYS_FIRST_DRAWING'] = np.nan
        self.X.loc[self.X['DAYS_FIRST_DUE'] == 365243, 'DAYS_FIRST_DUE'] = np.nan
        self.X.loc[self.X['DAYS_LAST_DUE_1ST_VERSION'] == 365243, 'DAYS_LAST_DUE_1ST_VERSION'] = np.nan
        self.X.loc[self.X['DAYS_LAST_DUE'] == 365243, 'DAYS_LAST_DUE'] = np.nan
        self.X.loc[self.X['DAYS_TERMINATION'] == 365243, 'DAYS_TERMINATION'] = np.nan

        self.X.rename(columns={'DAYS_LAST_DUE_1ST_VERSION': 'DAYS_LAST_DUE_V1'}, inplace=True)

        return self.X

    def ap_annuity_min(self):

        ap_group = self.X.groupby(['APPLICATION_NUMBER'])

        ap_group = ap_group.agg({'AMOUNT_ANNUITY': ['min']})
        ap_group.columns = [f"{feature}_{stat}" for feature, stat in ap_group]
        self.X = ap_group.reset_index()

        return self.X

    def ap_annuity_approved_min(self):

        status = 'Approved'
        agg_ap = self.X[self.X['NAME_CONTRACT_STATUS'] == status]

        ap_group = agg_ap.groupby(['APPLICATION_NUMBER'])

        ap_group = ap_group.agg({'AMOUNT_ANNUITY': ['min']})
        ap_group.columns = [f"{feature}_{status}_{stat}" for feature, stat in ap_group]
        self.X = ap_group.reset_index()

        return self.X

    def ap_annuity_max(self):

        ap_group = self.X.groupby(['APPLICATION_NUMBER'])

        ap_group = ap_group.agg({'AMOUNT_ANNUITY': ['max']})
        ap_group.columns = [f"{feature}_{stat}" for feature, stat in ap_group]
        self.X = ap_group.reset_index()

        return self.X

    def ap_annuity_approved_max(self):

        status = 'Approved'
        agg_ap = self.X[self.X['NAME_CONTRACT_STATUS'] == status]

        ap_group = agg_ap.groupby(['APPLICATION_NUMBER'])

        ap_group = ap_group.agg({'AMOUNT_ANNUITY': ['max']})
        ap_group.columns = [f"{feature}_{status}_{stat}" for feature, stat in ap_group]
        self.X = ap_group.reset_index()

        return self.X

    def ap_annuity_sum(self):

        ap_group = self.X.groupby(['APPLICATION_NUMBER'])

        ap_group = ap_group.agg({'AMOUNT_ANNUITY': ['sum']})
        ap_group.columns = [f"{feature}_{stat}" for feature, stat in ap_group]
        self.X = ap_group.reset_index()

        return self.X

    def ap_annuity_approved_sum(self):

        status = 'Approved'
        agg_ap = self.X[self.X['NAME_CONTRACT_STATUS'] == status]

        ap_group = agg_ap.groupby(['APPLICATION_NUMBER'])

        ap_group = ap_group.agg({'AMOUNT_ANNUITY': ['sum']})
        ap_group.columns = [f"{feature}_{status}_{stat}" for feature, stat in ap_group]
        self.X = ap_group.reset_index()

        return self.X

    def ap_credit_min(self):

        ap_group = self.X.groupby(['APPLICATION_NUMBER'])

        ap_group = ap_group.agg({'AMOUNT_CREDIT': ['min']})
        ap_group.columns = [f"{feature}_{stat}" for feature, stat in ap_group]
        self.X = ap_group.reset_index()

        return self.X

    def ap_credit_approved_min(self):

        status = 'Approved'
        agg_ap = self.X[self.X['NAME_CONTRACT_STATUS'] == status]

        ap_group = agg_ap.groupby(['APPLICATION_NUMBER'])

        ap_group = ap_group.agg({'AMOUNT_CREDIT': ['min']})
        ap_group.columns = [f"{feature}_{status}_{stat}" for feature, stat in ap_group]
        self.X = ap_group.reset_index()

        return self.X

    def ap_credit_max(self):

        ap_group = self.X.groupby(['APPLICATION_NUMBER'])

        ap_group = ap_group.agg({'AMOUNT_CREDIT': ['max']})
        ap_group.columns = [f"{feature}_{stat}" for feature, stat in ap_group]
        self.X = ap_group.reset_index()

        return self.X

    def ap_credit_approved_max(self):

        status = 'Approved'
        agg_ap = self.X[self.X['NAME_CONTRACT_STATUS'] == status]

        ap_group = agg_ap.groupby(['APPLICATION_NUMBER'])

        ap_group = ap_group.agg({'AMOUNT_CREDIT': ['max']})
        ap_group.columns = [f"{feature}_{status}_{stat}" for feature, stat in ap_group]
        self.X = ap_group.reset_index()

        return self.X

    def ap_credit_sum(self):

        ap_group = self.X.groupby(['APPLICATION_NUMBER'])

        ap_group = ap_group.agg({'AMOUNT_CREDIT': ['sum']})
        ap_group.columns = [f"{feature}_{stat}" for feature, stat in ap_group]
        self.X = ap_group.reset_index()

        return self.X

    def ap_credit_approved_sum(self):

        status = 'Approved'
        agg_ap = self.X[self.X['NAME_CONTRACT_STATUS'] == status]

        ap_group = agg_ap.groupby(['APPLICATION_NUMBER'])

        ap_group = ap_group.agg({'AMOUNT_CREDIT': ['sum']})
        ap_group.columns = [f"{feature}_{status}_{stat}" for feature, stat in ap_group]
        self.X = ap_group.reset_index()

        return self.X

    def ap_payment_min(self):

        ap_group = self.X.groupby(['APPLICATION_NUMBER'])

        ap_group = ap_group.agg({'AMOUNT_PAYMENT': ['min']})
        ap_group.columns = [f"{feature}_{stat}" for feature, stat in ap_group]
        self.X = ap_group.reset_index()

        return self.X

    def ap_payment_approved_min(self):

        status = 'Approved'
        agg_ap = self.X[self.X['NAME_CONTRACT_STATUS'] == status]

        ap_group = agg_ap.groupby(['APPLICATION_NUMBER'])

        ap_group = ap_group.agg({'AMOUNT_PAYMENT': ['min']})
        ap_group.columns = [f"{feature}_{status}_{stat}" for feature, stat in ap_group]
        self.X = ap_group.reset_index()

        return self.X

    def ap_payment_max(self):

        ap_group = self.X.groupby(['APPLICATION_NUMBER'])

        ap_group = ap_group.agg({'AMOUNT_PAYMENT': ['max']})
        ap_group.columns = [f"{feature}_{stat}" for feature, stat in ap_group]
        self.X = ap_group.reset_index()

        return self.X

    def ap_payment_approved_max(self):

        status = 'Approved'
        agg_ap = self.X[self.X['NAME_CONTRACT_STATUS'] == status]

        ap_group = agg_ap.groupby(['APPLICATION_NUMBER'])

        ap_group = ap_group.agg({'AMOUNT_PAYMENT': ['max']})
        ap_group.columns = [f"{feature}_{status}_{stat}" for feature, stat in ap_group]
        self.X = ap_group.reset_index()

        return self.X

    def ap_payment_sum(self):

        ap_group = self.X.groupby(['APPLICATION_NUMBER'])

        ap_group = ap_group.agg({'AMOUNT_PAYMENT': ['sum']})
        ap_group.columns = [f"{feature}_{stat}" for feature, stat in ap_group]
        self.X = ap_group.reset_index()

        return self.X

    def ap_payment_approved_sum(self):

        status = 'Approved'
        agg_ap = self.X[self.X['NAME_CONTRACT_STATUS'] == status]

        ap_group = agg_ap.groupby(['APPLICATION_NUMBER'])

        ap_group = ap_group.agg({'AMOUNT_PAYMENT': ['sum']})
        ap_group.columns = [f"{feature}_{status}_{stat}" for feature, stat in ap_group]
        self.X = ap_group.reset_index()

        return self.X

    # ***************

    def bk_credit_min(self):

        ap_group = self.X.groupby(['APPLICATION_NUMBER'])

        ap_group = ap_group.agg({'AMT_CREDIT_SUM': ['min']})
        ap_group.columns = [f"{feature}_{stat}" for feature, stat in ap_group]
        self.X = ap_group.reset_index()

        return self.X

    def bk_credit_active_min(self):

        credit = 'Active'
        agg_ap = self.X[self.X['CREDIT_ACTIVE'] == credit]

        ap_group = agg_ap.groupby(['APPLICATION_NUMBER'])

        ap_group = ap_group.agg({'AMT_CREDIT_SUM': ['min']})
        ap_group.columns = [f"{feature}_{credit}_{stat}" for feature, stat in ap_group]
        self.X = ap_group.reset_index()

        return self.X

    def bk_credit_max(self):

        ap_group = self.X.groupby(['APPLICATION_NUMBER'])

        ap_group = ap_group.agg({'AMT_CREDIT_SUM': ['max']})
        ap_group.columns = [f"{feature}_{stat}" for feature, stat in ap_group]
        self.X = ap_group.reset_index()

        return self.X

    def bk_credit_active_max(self):

        credit = 'Active'
        agg_ap = self.X[self.X['CREDIT_ACTIVE'] == credit]

        ap_group = agg_ap.groupby(['APPLICATION_NUMBER'])

        ap_group = ap_group.agg({'AMT_CREDIT_SUM': ['max']})
        ap_group.columns = [f"{feature}_{credit}_{stat}" for feature, stat in ap_group]
        self.X = ap_group.reset_index()

        return self.X

    def bk_credit_sum(self):

        ap_group = self.X.groupby(['APPLICATION_NUMBER'])

        ap_group = ap_group.agg({'AMT_CREDIT_SUM': ['sum']})
        ap_group.columns = [f"{feature}_{stat}" for feature, stat in ap_group]
        self.X = ap_group.reset_index()

        return self.X

    def bk_credit_active_sum(self):

        credit = 'Active'
        agg_ap = self.X[self.X['CREDIT_ACTIVE'] == credit]

        ap_group = agg_ap.groupby(['APPLICATION_NUMBER'])

        ap_group = ap_group.agg({'AMT_CREDIT_SUM': ['sum']})
        ap_group.columns = [f"{feature}_{credit}_{stat}" for feature, stat in ap_group]
        self.X = ap_group.reset_index()

        return self.X
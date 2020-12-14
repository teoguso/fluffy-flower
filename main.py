#!/usr/bin/env python
import logging

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, roc_curve, auc, plot_precision_recall_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

ORDER_DATA_PATH = Path("data/machine_learning_challenge_order_data.csv.gz")
LABEL_DATA_PATH = Path("data/machine_learning_challenge_labeled_data.csv.gz")
TYPED_ORDER_DATA_PATH = Path("data/order_data_batch.json")
CUSTOMER_FEATURES_PATH = Path("data/customer_features.json")
BEST_CUSTOMER_MODEL_PATH = Path("models/best_customer_model.joblib")
CUSTOMER_MODEL_ROC_PLOT_PATH = Path("data/customer_features_roc.png")
CUSTOMER_MODEL_PDIST_PRC_PLOT_PATH = Path("data/customer_features_pdist_prc.png")




def main():
    # Initial data processing
    preprocess_data()
    # Feature extraction
    create_customer_features()
    # Model training
    # "Customer features"
    ml_model_customer_features()


def ml_model_customer_features(force=False):
    if BEST_CUSTOMER_MODEL_PATH.exists() and not force:
        logger.warning("ML training skipped!")
    else:
        train, test = prepare_train_test()
        X = train.drop(columns=['is_returning_customer']).to_numpy()
        y = train['is_returning_customer'].to_numpy()
        logger.debug("Pipeline setup...")
        scaler = RobustScaler()
        rf = RandomForestClassifier(n_jobs=-1, verbose=1)
        logreg = LogisticRegression(n_jobs=-1, verbose=1)
        pipe = Pipeline(
            steps=[('scaler', scaler), ('classifier', rf)]
        )
        # Grid-search parameters
        param_grid = [
            {'classifier': [rf], 'classifier__n_estimators': [800]},
            {'classifier': [logreg], 'classifier__C': np.logspace(-3, 0, 3)},
        ]
        # Grid search definition
        search = GridSearchCV(pipe, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=3)
        logger.debug(f"Fitting grid search...")
        search.fit(X, y)
        logger.debug(f"Best model and parameters (CV score={search.best_score_:.3f}):")
        logger.debug(f"{search.best_params_}")
        logger.debug(f"Best Estimator:")
        logger.debug(f"{search.best_estimator_}")
        with open(BEST_CUSTOMER_MODEL_PATH, 'wb') as file_handler:
            joblib.dump(search, file_handler)
        x_test = test.drop(columns=['is_returning_customer'])
        y_test = test['is_returning_customer'].to_numpy()
        print(classification_report(y_test, search.predict(x_test)))
        y_proba = search.predict_proba(x_test)
        # Plotting
        plt.style.use('ggplot')
        f1, auc_score = plot_roc_auc_f1(y_test, y_proba)
        plt.savefig(CUSTOMER_MODEL_ROC_PLOT_PATH, dpi=150)
        logger.debug(f"AUC (ROC): {auc_score}")
        logger.debug(f"f1 score: {f1}")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        plot_probability_distribution(y_proba, y_test, ax1)
        plot_precision_recall_curve(search, x_test, y_test, ax=ax2)
        ax2.set_title("Precision-Recall curve")
        plt.savefig(CUSTOMER_MODEL_PDIST_PRC_PLOT_PATH, dpi=150)


def prepare_train_test():
    logger.debug(f"Reading customer features from {CUSTOMER_FEATURES_PATH}...")
    customer_features = pd.read_json(
        CUSTOMER_FEATURES_PATH,
        orient='table',
    )
    labels = pd.read_csv(LABEL_DATA_PATH)
    # This aligns the labels with the features on the correct CID
    labels = customer_features.join(labels.set_index('customer_id'))['is_returning_customer']
    features_plus_labels = customer_features.join(labels)
    # NOTE: This is cheating because we already know which one is the majority class
    logger.debug(f"Balancing classes...")
    negative_data = features_plus_labels.loc[labels == 0].sample(
        n=(labels == 1).sum(),
        random_state=42,
    )
    positive_data = features_plus_labels.loc[labels == 1]
    balanced_data = pd.concat((positive_data, negative_data))
    logger.debug(f"Positive/negative class ratio: {balanced_data['is_returning_customer'].mean()}")
    logger.debug(f"Splitting train/test...")
    # Train/test split
    train, test = train_test_split(
        balanced_data,
        train_size=.8,
        shuffle=True,
        stratify=balanced_data['is_returning_customer'].to_numpy(),
        random_state=42,
    )
    return train, test


def plot_probability_distribution(y_proba, y_true, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    pd.Series(y_proba[:, 1]).loc[y_true >= .5].plot.hist(bins=99, alpha=.5, label='Positive', ax=ax)
    pd.Series(y_proba[:, 1]).loc[y_true < .5].plot.hist(bins=99, alpha=.5, label='Negative', ax=ax)
    ax.set_xlabel("Probability")
    ax.set_title("Probability Distribution")
    ax.legend()


def plot_roc_auc_f1(true_labels, probability, title=None):
    # This is a convenience function that takes care of boring stuff
    f1 = f1_score(true_labels, probability[:, 1]>.5)
    fpr, tpr, _ = roc_curve(true_labels, probability[:, 1])
    auc_score = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 6))
    if title is not None:
        ax.set_title(title)
    ax.plot([0, 1], [0, 1], '--', label="Random")
    ax.plot(fpr, tpr, label="Your model")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.annotate(f"AUC: {auc_score:.4}", (.8, 0.15))
    ax.annotate(f"F1: {f1:.4}", (.8, 0.1))
    ax.legend()
    return f1, auc_score


def create_customer_features(force=False):
    if CUSTOMER_FEATURES_PATH.exists() and not force:
        logger.warning("Features creation skipped!")
    else:
        logger.debug("Reading typed data...")
        df_orders = pd.read_json(
            TYPED_ORDER_DATA_PATH,
            orient='table',
        )
        logger.debug("Creating time-based order features..")
        # Holidays
        df_orders['is_holiday'] = (
                (
                        (df_orders['order_datetime'].dt.month == 1) & (df_orders['order_datetime'].dt.day == 1)
                ) | (
                        (df_orders['order_datetime'].dt.month == 12) & (
                        (df_orders['order_datetime'].dt.day == 25) | (df_orders['order_datetime'].dt.day == 31))
                )
        )
        # Time of day, day of week
        df_orders['hour_of_day'] = df_orders['order_datetime'].dt.hour
        df_orders['day_of_week'] = df_orders['order_datetime'].dt.dayofweek
        # Time-based features
        logger.debug("Creating customer features..")
        last_orders = df_orders.groupby('customer_id')['order_datetime'].max()
        max_datetime = df_orders['order_datetime'].max()
        last_order_age_days = (max_datetime - last_orders).dt.days
        last_order_age_days.name = 'last_order_age_days'
        customer_features = last_order_age_days.to_frame()
        n_orders_holidays = df_orders.groupby('customer_id')['is_holiday'].sum()
        n_orders_holidays.name = 'n_orders_holidays'
        customer_features = customer_features.join(n_orders_holidays)
        mean_hour_of_day = df_orders.groupby('customer_id')['hour_of_day'].mean()
        mean_hour_of_day.name = 'mean_hour_of_day'
        customer_features = customer_features.join(mean_hour_of_day)
        std_hour_of_day = df_orders.groupby('customer_id')['hour_of_day'].std().fillna(0)
        std_hour_of_day.name = 'std_hour_of_day'
        customer_features = customer_features.join(std_hour_of_day)
        mean_day_of_week = df_orders.groupby('customer_id')['day_of_week'].mean()
        mean_day_of_week.name = 'mean_day_of_week'
        customer_features = customer_features.join(mean_day_of_week)
        std_day_of_week = df_orders.groupby('customer_id')['day_of_week'].std().fillna(0)
        std_day_of_week.name = 'std_day_of_week'
        customer_features = customer_features.join(std_day_of_week)
        # Orders features
        first_orders = df_orders.groupby('customer_id')['order_datetime'].min()
        first_order_age_days = (max_datetime - first_orders).dt.days
        first_order_age_days.name = 'first_order_age_days'
        customer_features = customer_features.join(first_order_age_days)
        # Amount paid
        number_of_orders_per_customer = df_orders.groupby('customer_id')['amount_paid'].count()
        number_of_orders_per_customer.name = 'n_orders'
        customer_features = customer_features.join(number_of_orders_per_customer)
        # Customer order rank
        max_customer_order_rank = df_orders.groupby(
            'customer_id'
        )['customer_order_rank'].max()
        max_customer_order_rank
        max_customer_order_rank.name = 'max_customer_order_rank'
        customer_features = customer_features.join(max_customer_order_rank)
        # Failed orders
        failed_orders_per_customer = df_orders.groupby('customer_id')['is_failed'].sum()
        failed_orders_per_customer.name = 'n_failed'
        customer_features = customer_features.join(failed_orders_per_customer)
        # Voucher amount
        max_voucher_amount = df_orders.groupby('customer_id')['voucher_amount'].max()
        max_voucher_amount.name = 'max_voucher_amount'
        customer_features = customer_features.join(max_voucher_amount)
        tot_voucher_amount = df_orders.groupby('customer_id')['voucher_amount'].sum()
        tot_voucher_amount.name = 'tot_voucher_amount'
        customer_features = customer_features.join(tot_voucher_amount)
        n_vouchers = (df_orders['voucher_amount'] > 0).groupby(df_orders['customer_id']).sum()
        n_vouchers.name = 'n_vouchers'
        customer_features = customer_features.join(n_vouchers)
        # Delivery fee
        tot_delivery_fee = df_orders.groupby('customer_id')['delivery_fee'].sum()
        max_delivery_fee = df_orders.groupby('customer_id')['delivery_fee'].max()
        # How many times a delivery fee was paid
        n_delivery_fee = (df_orders['delivery_fee'] > 0).groupby(df_orders['customer_id']).sum()
        tot_delivery_fee.name = 'tot_delivery_fee'
        max_delivery_fee.name = 'max_delivery_fee'
        n_delivery_fee.name = 'n_delivery_fee'
        customer_features = customer_features.join(tot_delivery_fee)
        customer_features = customer_features.join(max_delivery_fee)
        customer_features = customer_features.join(n_delivery_fee)
        # Amount paid
        tot_amount_paid = df_orders.groupby('customer_id')['amount_paid'].sum()
        avg_amount_paid = df_orders.groupby('customer_id')['amount_paid'].mean()
        max_amount_paid = df_orders.groupby('customer_id')['amount_paid'].max()
        min_amount_paid = df_orders.groupby('customer_id')['amount_paid'].min()
        tot_amount_paid.name = 'tot_amount_paid'
        avg_amount_paid.name = 'avg_amount_paid'
        max_amount_paid.name = 'max_amount_paid'
        min_amount_paid.name = 'min_amount_paid'
        customer_features = customer_features.join(tot_amount_paid)
        customer_features = customer_features.join(avg_amount_paid)
        customer_features = customer_features.join(max_amount_paid)
        customer_features = customer_features.join(min_amount_paid)
        # Restaurants
        n_restaurants = df_orders.groupby('customer_id')['restaurant_id'].nunique()
        n_restaurants.name = 'n_restaurants'
        customer_features = customer_features.join(n_restaurants)
        # Cities
        n_cities = df_orders.groupby('customer_id')['city_id'].nunique()
        n_cities.name = 'n_cities'
        customer_features = customer_features.join(n_cities)
        # Store to disk
        logger.debug(f"Saving to disk at {CUSTOMER_FEATURES_PATH.as_posix()}...")
        customer_features.to_json(
            CUSTOMER_FEATURES_PATH,
            orient='table',
        )


def preprocess_data(force=False):
    """Produce a correctly typed dataset from raw data, if necessary"""
    if TYPED_ORDER_DATA_PATH.exists() and not force:
        logger.warning("Preprocessing skipped!")
    else:
        # Read data
        logger.debug(f"Reading {ORDER_DATA_PATH}")
        order_data = pd.read_csv(ORDER_DATA_PATH)
        # Impute missing values
        logger.debug(f"Imputing missing data...")
        order_data.fillna(0, inplace=True)
        # ### Set proper data types
        logger.debug(f"Setting data types...")
        # Order rank
        order_data['customer_order_rank'] = order_data['customer_order_rank'].astype(np.int64)
        # Categories
        for col in order_data.columns:
            if col.endswith('_id') and order_data[col].dtype == 'int':
                order_data[col] = order_data[col].astype('str').astype('category')
            else:
                continue
        order_data['order_datetime'] = pd.to_datetime(
            order_data['order_date'] + ' ' + order_data['order_hour'].astype('str').str.pad(width=2, fillchar='0'),
            format='%Y-%m-%d %H',
        )
        # Cleaning up
        order_data.drop(columns=['order_date', 'order_hour'], inplace=True)
        # Remove sparse period
        order_data = order_data.set_index(
            'order_datetime'
        ).loc["2015-03-01":].reset_index()
        # Storing
        logger.debug(f"Storing typed data to {TYPED_ORDER_DATA_PATH}...")
        logger.debug(f"Data types: \n{order_data.dtypes}")
        with open(TYPED_ORDER_DATA_PATH, 'w') as file_handler:
            order_data.to_json(file_handler, orient='table')


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
    )
    main()

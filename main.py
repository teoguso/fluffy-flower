#!/usr/bin/env python
import logging

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder

from returning.eval import print_plot_metrics

logger = logging.getLogger(__name__)

ORDER_DATA_PATH = Path("data/machine_learning_challenge_order_data.csv.gz")
LABEL_DATA_PATH = Path("data/machine_learning_challenge_labeled_data.csv.gz")
TYPED_ORDER_DATA_PATH = Path("data/order_data_batch.json")
CUSTOMER_FEATURES_PATH = Path("data/customer_features.json")
DUMMY_FEATURES_PATH = Path("data/dummy_features.json")
BEST_CUSTOMER_MODEL_PATH = Path("models/best_customer_model.joblib")
BEST_DUMMY_MODEL_PATH = Path("models/best_dummy_model.joblib")
CUSTOMER_MODEL_ROC_PLOT_PATH = Path("data/customer_features_roc.png")
CUSTOMER_MODEL_PDIST_PREC_REC_PLOT_PATH = Path("data/customer_features_pdist_prc.png")
DUMMIES_MODEL_ROC_PLOT_PATH = Path("data/dummy_features_roc.png")
DUMMIES_MODEL_PDIST_PREC_REC_PLOT_PATH = Path("data/dummy_features_pdist_prc.png")


def main():
    # Initial data processing
    preprocess_data(ORDER_DATA_PATH, TYPED_ORDER_DATA_PATH)
    # Feature extraction
    create_customer_features(TYPED_ORDER_DATA_PATH, CUSTOMER_FEATURES_PATH)
    # Model training
    # "Customer features"
    df_train, df_test = prepare_train_test(CUSTOMER_FEATURES_PATH, LABEL_DATA_PATH)
    search_grid_fit_cf = ml_model_customer_features(training_data=df_train, trained_model_path=BEST_CUSTOMER_MODEL_PATH)
    print_plot_metrics(
        search_grid_fit_cf,
        df_test,
        roc_out_path=CUSTOMER_MODEL_ROC_PLOT_PATH,
        proba_dist_prec_rec_path=CUSTOMER_MODEL_PDIST_PREC_REC_PLOT_PATH,
        title="Customer Features Model",
    )
    # Dummy features model
    dummy_features = create_dummy_features(TYPED_ORDER_DATA_PATH, DUMMY_FEATURES_PATH)
    # Model training
    # df_train_dummy, df_test_dummy = prepare_train_test(DUMMY_FEATURES_PATH, LABEL_DATA_PATH)
    df_train_dummy, df_test_dummy = prepare_train_test(dummy_features, LABEL_DATA_PATH)
    search_grid_fit_dummy = ml_model_dummy_features(df_train_dummy, BEST_DUMMY_MODEL_PATH)
    print_plot_metrics(
        search_grid_fit_dummy,
        df_test_dummy,
        roc_out_path=DUMMIES_MODEL_ROC_PLOT_PATH,
        proba_dist_prec_rec_path=DUMMIES_MODEL_PDIST_PREC_REC_PLOT_PATH,
        title="Dummy Features Model",
    )


def create_dummy_features(typed_order_data_path, output_features_path, force=False):
    """Extract one-hot representation (AKA dummy variables) from categories

    :param typed_order_data_path:
    :param output_features_path:
    :param force:
    :return:
    """
    # if output_features_path.exists() and not force:
    #     logger.warning("Dummy features creation skipped!")
    # else:
    logger.debug("Reading typed data...")
    orders = pd.read_json(
        typed_order_data_path,
        orient='table',
    )
    # Time of day, day of week
    orders['hour_of_day'] = orders['order_datetime'].dt.hour
    orders['day_of_week'] = orders['order_datetime'].dt.dayofweek
    # categorical_columns = ['restaurant_id',
    #                        'city_id', 'payment_id', 'platform_id', 'transmission_id',
    #                        'hour_of_day', 'day_of_week']
    categorical_columns_miniset = [
        #     'restaurant_id',
        #        'city_id',
        'payment_id', 'platform_id', 'transmission_id',
        'hour_of_day', 'day_of_week']
    # other_columns = ['order_datetime', 'customer_order_rank', 'is_failed',
    #                  'voucher_amount', 'delivery_fee', 'amount_paid', 'is_holiday']
    # groupby_column = 'customer_id'
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit(orders[categorical_columns_miniset].to_numpy())
    logger.debug(f"Categories in the encoder: {encoder.categories_}")
    sparse_dummy_vars_miniset = pd.DataFrame(
        index=orders['customer_id'],
        data=encoder.transform(orders[categorical_columns_miniset]).A,
    )
    customer_dummy_features = sparse_dummy_vars_miniset.groupby(sparse_dummy_vars_miniset.index).sum()
    # Store to disk
    logger.debug(f"Saving to disk at {output_features_path.as_posix()}...")
    customer_dummy_features.to_json(
        output_features_path,
        orient='table',
    )
    return customer_dummy_features


def ml_model_dummy_features(training_data, trained_model_path, force=False):
    if trained_model_path.exists() and not force:
        logger.warning("ML training 'dummy features' skipped!")
        search = joblib.load(trained_model_path)
    else:
        X = training_data.drop(columns=['is_returning_customer']).to_numpy()
        y = training_data['is_returning_customer'].to_numpy()
        nb = MultinomialNB()
        logreg = LogisticRegression(n_jobs=-1, verbose=3)
        pipe = Pipeline(
            steps=[('classifier', nb)]
        )
        param_grid = [
            {'classifier': [nb], 'classifier__alpha': [.1, .3, .6, 1]},
            {'classifier': [logreg], 'classifier__C': np.logspace(-5, 0, 6)},
        ]
        search = GridSearchCV(pipe, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=3)
        logger.debug(f"Fitting grid search...")
        search.fit(X, y)
        logger.debug(f"Best model and parameters (CV score={search.best_score_:.3f}):")
        logger.debug(f"{search.best_params_}")
        logger.debug(f"Best Estimator:")
        logger.debug(f"{search.best_estimator_}")
        with open(trained_model_path, 'wb') as file_handler:
            joblib.dump(search, file_handler)
    return search


def ml_model_customer_features(training_data, trained_model_path, force=False):
    if trained_model_path.exists() and not force:
        logger.warning("ML training 'customer features' skipped!")
        search = joblib.load(trained_model_path)
    else:
        X = training_data.drop(columns=['is_returning_customer']).to_numpy()
        y = training_data['is_returning_customer'].to_numpy()
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
        with open(trained_model_path, 'wb') as file_handler:
            joblib.dump(search, file_handler)
    return search


def prepare_train_test(customer_features_path, label_data_path):
    logger.debug(f"Reading customer features from {customer_features_path}...")
    logger.debug(f"customer_features_path: {customer_features_path}")
    if isinstance(customer_features_path, Path):
        customer_features = pd.read_json(
            customer_features_path,
            orient='table',
        )
    else:
        # Read directly a dataframe
        customer_features = customer_features_path
    labels = pd.read_csv(label_data_path)
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


def create_customer_features(typed_order_data_path, customer_features_path, force=False):
    if customer_features_path.exists() and not force:
        logger.warning("Features creation skipped!")
    else:
        logger.debug("Reading typed data...")
        df_orders = pd.read_json(
            typed_order_data_path,
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
        logger.debug(f"Saving to disk at {customer_features_path.as_posix()}...")
        customer_features.to_json(
            customer_features_path,
            orient='table',
        )
        return customer_features


def preprocess_data(input_data_path, output_data_path, force=False):
    """Produce a correctly typed dataset from raw data, if necessary
    :param output_data_path:
    :param input_data_path:
    :param force:
    """
    if output_data_path.exists() and not force:
        logger.warning("Preprocessing skipped!")
    else:
        # Read data
        data_path = input_data_path
        logger.debug(f"Reading {data_path}")
        order_data = pd.read_csv(data_path)
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
        logger.debug(f"Storing typed data to {output_data_path}...")
        logger.debug(f"Data types: \n{order_data.dtypes}")
        with open(output_data_path, 'w') as file_handler:
            order_data.to_json(file_handler, orient='table')


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    main()

#!/usr/bin/env python
import logging
import numpy as np
import pandas as pd
from pathlib import Path


logger = logging.getLogger(__name__)

ORDER_DATA_PATH = Path("data/machine_learning_challenge_order_data.csv.gz")
LABEL_DATA_PATH = Path("data/machine_learning_challenge_labeled_data.csv.gz")
TYPED_ORDER_DATA_PATH = Path("data/order_data_batch.json")
CUSTOMER_FEATURES_PATH = Path("data/customer_features.json")


def main():
    # Initial data processing
    data_preprocessing()
    # ### FEATURE EXTRACTION
    create_customer_features()


def create_customer_features():
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


def data_preprocessing():
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

#!/usr/bin/env python
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from pyarrow import Table
from pyarrow import parquet as pq


logger = logging.getLogger(__name__)

ORDER_DATA_PATH = Path("data/machine_learning_challenge_order_data.csv.gz")
LABEL_DATA_PATH = Path("machine_learning_challenge_labeled_data.csv.gz")
TYPED_ORDER_DATA_PATH = Path("data/order_data_batch.json")


def main():
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
            order_data[col] = order_data[col].astype('category')
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
    print(order_data.dtypes)
    with open(TYPED_ORDER_DATA_PATH, 'w') as file_handler:
        order_data.to_json(file_handler, orient='table')
        # order_data.to_parquet(
        #     TYPED_ORDER_DATA,
        #     # partition_cols=['order_datetime']
        # )
    # ### FEATURE EXTRACTION


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
    )
    main()

#!/usr/bin/env python
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

ORDER_DATA = Path("data/machine_learning_challenge_order_data.csv.gz")
LABEL_DATA = Path("machine_learning_challenge_labeled_data.csv.gz")
TYPED_ORDER_DATA = Path("data/order_data.parquet")


def main():
    # Read data
    logger.debug(f"Reading {ORDER_DATA}")
    order_data = pd.read_csv(ORDER_DATA)
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
    # Storing
    logger.debug(f"Storing typed data to {TYPED_ORDER_DATA}...")
    order_data.to_parquet(
        TYPED_ORDER_DATA,
        partition_cols=['order_datetime']
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
    )
    main()

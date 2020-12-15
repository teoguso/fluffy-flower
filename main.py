#!/usr/bin/env python
import logging

from pathlib import Path

from returning.eval import print_plot_metrics
from returning.ml import create_dummy_features, ml_model_dummy_features, ml_model_customer_features, prepare_train_test, \
    create_customer_features, preprocess_data

logger = logging.getLogger('returning')

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
    # ### "Customer features" model
    # Feature extraction
    create_customer_features(TYPED_ORDER_DATA_PATH, CUSTOMER_FEATURES_PATH)
    # Model training
    df_train, df_test = prepare_train_test(CUSTOMER_FEATURES_PATH, LABEL_DATA_PATH)
    search_grid_fit_cf = ml_model_customer_features(training_data=df_train, trained_model_path=BEST_CUSTOMER_MODEL_PATH)
    print_plot_metrics(
        search_grid_fit_cf,
        df_test,
        roc_out_path=CUSTOMER_MODEL_ROC_PLOT_PATH,
        proba_dist_prec_rec_path=CUSTOMER_MODEL_PDIST_PREC_REC_PLOT_PATH,
        title="Customer Features Model",
    )
    # ### Dummy features model
    dummy_features = create_dummy_features(TYPED_ORDER_DATA_PATH, DUMMY_FEATURES_PATH)
    # Model training
    df_train_dummy, df_test_dummy = prepare_train_test(dummy_features, LABEL_DATA_PATH)
    search_grid_fit_dummy = ml_model_dummy_features(df_train_dummy, BEST_DUMMY_MODEL_PATH)
    print_plot_metrics(
        search_grid_fit_dummy,
        df_test_dummy,
        roc_out_path=DUMMIES_MODEL_ROC_PLOT_PATH,
        proba_dist_prec_rec_path=DUMMIES_MODEL_PDIST_PREC_REC_PLOT_PATH,
        title="Dummy Features Model",
    )


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    main()

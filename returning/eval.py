import logging
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, plot_precision_recall_curve, f1_score, roc_curve, auc


logger = logging.getLogger(__name__)


def print_plot_metrics(fit_search_grid, test_data, roc_out_path, proba_dist_prec_rec_path, title=None):
    x_test = test_data.drop(columns=['is_returning_customer'])
    y_test = test_data['is_returning_customer'].to_numpy()
    print(classification_report(y_test, fit_search_grid.predict(x_test)))
    y_proba = fit_search_grid.predict_proba(x_test)
    # Plotting
    plt.style.use('ggplot')
    f1, auc_score = plot_roc_auc_f1(y_test, y_proba, title=title)
    plt.savefig(roc_out_path, dpi=150)
    logger.debug(f"AUC (ROC): {auc_score}")
    logger.debug(f"f1 score: {f1}")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(title)
    plot_probability_distribution(y_proba, y_test, ax1)
    plot_precision_recall_curve(fit_search_grid, x_test, y_test, ax=ax2)
    ax2.set_title("Precision-Recall curve")
    plt.savefig(proba_dist_prec_rec_path, dpi=150)


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
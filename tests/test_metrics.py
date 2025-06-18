from neuralk_foundry_ce.utils.metrics import *


def test_metrics():
    import numpy as np

    # Setup dummy data
    y_true_bin = np.array([0, 1, 1, 0, 1])
    y_pred_bin = np.array([0, 1, 0, 0, 1])
    y_score_bin = np.array([[0.8, 0.2], [0.2, 0.8], [0.6, 0.4], [0.7, 0.3], [0.1, 0.9]])

    y_true_multi = np.array([0, 2, 1, 2, 1])
    y_pred_multi = np.array([0, 2, 1, 0, 1])
    y_score_multi = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.3, 0.6],
        [0.1, 0.8, 0.1],
        [0.6, 0.2, 0.2],
        [0.2, 0.7, 0.1]
    ])

    y_true_reg = np.array([2.5, 0.0, 2.1, 1.6])
    y_pred_reg = np.array([3.0, -0.1, 2.0, 1.5])

    X_clust = np.random.rand(10, 5)
    y_true_clust = np.array([0, 1, 1, 0, 1, 2, 2, 0, 0, 2])
    y_pred_clust = np.array([0, 1, 1, 0, 1, 1, 2, 0, 0, 2])

    y_true_link = np.array([0, 0, 1, 1, 2])
    all_pairs = [(0, 1), (0, 2), (1, 2), (3, 4)]
    y_pairs = [1, 0, 0, 0]

    y_group = np.array([1, 1, 2, 3, 3])
    y_group_pred = np.array([4, 4, 5, 6, 6])

    # Instantiate all metrics to test
    metrics = [
        CrossEntropy(), AccuracyScore(), KLDivergence(), Recall(),
        Precision(), F1ScoreMetric(), HingeLossMetric(), ROCAUC(),
        L1Loss(), L2Loss(), HuberLoss(), RSquared(),
        Silhouette(), AdjustedRandIndex(), NormalizedMutualInfo(),
        LinkageAveragePrecision(), LinkageAverageRecall(),
        JaccardSimilarity(), GroupDetectionPrecision(), GroupDetectionRecall()
    ]

    print("Testing metrics:")
    for metric in metrics:
        name = metric.__class__.__name__
        try:
            if isinstance(metric, CrossEntropy) or isinstance(metric, HingeLossMetric) or isinstance(metric, ROCAUC):
                result = metric(y_true_bin, y_pred_bin, y_score=y_score_bin)
            elif isinstance(metric, KLDivergence):
                result = metric(y_true=y_score_bin, y_pred=y_score_bin + 1e-6)  # avoid zero division
            elif isinstance(metric, Silhouette):
                result = metric(y_true=X_clust, y_pred=y_pred_clust)
            elif isinstance(metric, (L1Loss, L2Loss, HuberLoss, RSquared)):
                result = metric(y_true=y_true_reg, y_pred=y_pred_reg)
            elif isinstance(metric, (LinkageAveragePrecision, LinkageAverageRecall)):
                result = metric(y_true=y_true_link, y_pred=None, y_pairs=y_pairs, all_pairs=all_pairs)
            elif isinstance(metric, (GroupDetectionPrecision, GroupDetectionRecall)):
                result = metric(y_true=y_group, y_pred=y_group_pred)
            else:
                result = metric(y_true=y_true_bin, y_pred=y_pred_bin)
        except Exception as e:
            result = f"Error: {str(e)}"
        print(f"{name:30s}: {result:.4f}" if isinstance(result, float) else f"{name:30s}: {result}")

# Author Tuka Alhanai SLS CSAIL MIT May 2018


import numpy as np
from sklearn import metrics
from scipy import stats


# ======================================================================================================================
# GET PERFORMANCE METRICS
# ======================================================================================================================
def perfMetrics(pred, Y):
    """
        Evaluates performance of classifier given probs and true labels.
    :param pred: predicted probabilities
    :param Y: true labels
    :return: dict with results
    """
    auc = metrics.roc_auc_score(Y, pred)

    fpr, tpr, thresholds = metrics.roc_curve(Y, pred, pos_label=1)
    tpr_fpr0 = np.max(tpr[np.where(fpr <= 0.00)])
    tpr_fpr1 = np.max(tpr[np.where(fpr <= 0.01)])
    tpr_fpr5 = np.max(tpr[np.where(fpr <= 0.05)])

    f1_bin = metrics.f1_score(Y, np.round(pred), pos_label=1)
    f1_micro = metrics.f1_score(Y, np.round(pred), average='micro', pos_label=1)
    f1_macro = metrics.f1_score(Y, np.round(pred), average='macro', pos_label=1)

    [p, r, th] = metrics.precision_recall_curve(Y, pred)
    auc_prc = metrics.auc(r, p)
    prec = metrics.precision_score(Y, np.round(pred))
    rec = metrics.recall_score(Y, np.round(pred))

    acc = metrics.accuracy_score(Y, np.round(pred), normalize=False)
    acc_norm = metrics.accuracy_score(Y, np.round(pred), normalize=True)

    brier = metrics.brier_score_loss(Y, pred)

    hl = hltest(Y, pred)

    m = dict()
    m['auc'] = auc
    m['f1_bin'], m['f1_micro'], m['f1_macro'] = f1_bin, f1_micro, f1_macro
    m['tpr_fpr0'], m['tpr_fpr1'], m['tpr_fpr5'] = tpr_fpr0, tpr_fpr1, tpr_fpr5
    m['auc_prc'], m['prec'], m['rec'] = auc_prc, prec, rec
    m['acc'], m['acc_norm'] = acc, acc_norm
    m['brier'], m['hltest'] = brier, hl

    return m



# ======================================================================================================================
# HOSMER-LEMESHOW TEST
# ======================================================================================================================
def hltest(Y_true, Y_pred, Nbins=10):
    """
    Calculating the Hosmer-Lemeshow Test for statistical callibration.
    Useful example: https://en.wikipedia.org/wiki/Hosmer%E2%80%93Lemeshow_test#Calculation_of_the_statistic

    :param Y_true: 1/0 binary class of true outcomes
    :param Y_pred: predicted probability of outcome of 1
    :return: pval chisquare statistical significance of model callibration (pval > 0.05 means model is well-calibrated)
    """

    # convert to numpy array
    Y_true = np.asarray(Y_true)
    Y_pred = np.asarray(Y_pred)

    # split probs into 10 bins
    pred_min = np.min(Y_pred)
    pred_max = np.max(Y_pred)

    # Nbins = 10
    prob_int = (pred_max - pred_min) / Nbins

    # calucalte HL-stat for 10 bins
    H = []
    for i in range(Nbins):

        # N observations that were correct in this interval
        obsA = np.where((Y_pred >= (pred_min + i*prob_int))
                        & (Y_pred < (pred_min + (i+1)*prob_int))
                        & (Y_true == 1))[0].shape[0]

        # N observations that were NOT correct in this interval
        obsNotA = np.where((Y_pred >= (pred_min + i * prob_int))
                        & (Y_pred < (pred_min + (i + 1) * prob_int))
                        & (Y_true != 1))[0].shape[0]

        # index of probs we are considering in this interval
        idx_A = np.where((Y_pred >= (pred_min + i * prob_int))
                        & (Y_pred < (pred_min + (i + 1) * prob_int)))[0]

        # calculate the sum of the probabilities
        expA = np.sum(Y_pred[idx_A])

        # calculate the total sum minus the sum of the predicted probabilities
        expNotA = idx_A.shape[0] - expA

        # calculate HL-stat for this bin/interval
        H.append((obsA - expA) ** 2 / expA + (obsNotA - expNotA) ** 2 / expNotA)


    # sum and calculate the statistical significance that the model follows a chi-distributions (i.e. good-fit, well-callibrated)
    H = np.nansum(H)
    pval = 1 - stats.chi2.cdf(x=H, df=Nbins-2)

    # print result
    # if pval > 0.05:
    #     print('   Model is WELL calibrated: ' + str(pval))
    # else:
    #     print('   Model is POORLY calibrated: ' + str(pval))

    return pval

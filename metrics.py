import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


class Metrics():

    def __init__(self, nclass=7):
        self.nclass = nclass
        self.confusion_matrix = np.zeros((nclass, nclass))

    def metrics(self):
        # l_pred, l_true = self.use_mask(mask ,l_pred, l_true)
        hist = self.confusion_matrix
        conf = ConfusionMatrixDisplay(hist)
        conf.plot()
        # conf.show()
        # plt.show()
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.nclass), iu))
        return {
                "Overall Acc:": acc,
                "Mean Acc:": acc_cls,
                "FreqW Acc:": fwavacc,
                "Mean IoU:": mean_iu,
                "Classes": cls_iu
                }

    @staticmethod
    def use_mask(mask, lpred, ltrue):
        mask = mask > 0
        return lpred[mask], ltrue[mask]

    def hist(self, l_pred, l_true):
        hist = np.bincount(
            self.nclass * l_true.astype(int) + l_pred, minlength=self.nclass ** 2
        ).reshape(self.nclass, self.nclass)
        return hist

    def update(self, l_pred, l_true):
        for lp, lt in zip(l_pred, l_true):
            self.confusion_matrix += self.hist(lp.flatten(), lt.flatten())

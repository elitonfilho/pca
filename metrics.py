import numpy as np

class Metrics():

    def __init__(self, nbands=6, nclass=19):
        self.nbands = nbands
        self.nclass = nclass
        self.confusion_matrix = np.zeros((nclass, nclass))

    def metrics(self,mask, l_pred, l_true):
        l_pred, l_true = self.use_mask(mask ,l_pred, l_true)
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.nclass), iu))
        return (
            {
                "Overall Acc: \t": acc,
                "Mean Acc : \t": acc_cls,
                "FreqW Acc : \t": fwavacc,
                "Mean IoU : \t": mean_iu,
            },
            cls_iu,
        )

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

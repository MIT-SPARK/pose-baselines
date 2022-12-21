
import pickle5 as pickle
import numpy as np


def get_auc(rec, threshold):

    rec = np.sort(rec)
    rec = np.where(rec <= threshold, rec, np.array([float("inf")]))
    # print(rec)
    # print(rec.shape)
    # breakpoint()

    n = rec.shape[0]
    prec = np.cumsum(np.ones(n) / n, axis=0)

    index = np.isfinite(rec)
    rec = rec[index]
    prec = prec[index]

    if len(rec) == 0:
        # print("returns zero: ", 0.0)
        return np.asarray([0.0])[0]
    else:
        # print(prec)
        # print(prec.shape)
        mrec = np.zeros(rec.shape[0] + 2)
        mrec[0] = 0
        mrec[-1] = threshold
        mrec[1:-1] = rec

        mpre = np.zeros(prec.shape[0] + 2)
        mpre[1:-1] = prec
        mpre[-1] = prec[-1]

        for i in range(1, mpre.shape[0]):
            mpre[i] = max(mpre[i], mpre[i - 1])

        ap = 0
        ap = np.zeros(1)
        for i in range(mrec.shape[0] - 1):
            # print("mrec[i+1] ", mrec[i+1])
            # print("mpre[i+1] ", mpre[i+1])
            # ap += (mrec[i+1] - mrec[i]) * mpre[i+1]
            ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1] * (1 / threshold)

        # print(ap)
        # print(type(ap))
        # print("returns ap: ", ap[0])
        # breakpoint()
        return ap[0]


class EvalData:
    def __init__(self, adds_th=0.02, adds_auc_th=0.05):
        self.eval_store_metrics = ["adds", "oc", "nd", \
                                   "rerr", "terr", \
                                   "adds_oc", "adds_oc_nd", \
                                   "adds_auc", "adds_oc_auc", "adds_oc_nd_auc", \
                                   "adds_th_score", "adds_oc_th_score", "adds_oc_nd_th_score", \
                                   "adds_th", "adds_auc_th"]

        self.data = dict()
        for metric in self.eval_store_metrics:
            self.data[metric] = None

        self.n = None
        self.data['adds_th'] = adds_th
        self.data['adds_auc_th'] = adds_auc_th

    def set_adds(self, adds_):
        self.data["adds"] = adds_
        self.n = len(adds_)

    def set_oc(self, oc_):
        self.data["oc"] = oc_

    def set_nd(self, nd_):
        self.data["nd"] = nd_

    def set_rerr(self, rerr_):
        self.data["rerr"] = rerr_
        self.n = len(rerr_)

    def set_terr(self, terr_):
        self.data["terr"] = terr_
        self.n = len(terr_)

    def set_adds_th(self, th_):
        self.data["adds_th"] = th_

    def set_adds_auc_th(self, th_):
        self.data["adds_auc_th"] = th_

    def complete_eval_data(self):

        # breakpoint()
        if self.n is None:
            self.n = len(self.data["adds"])

        # if oc or nd is None, we fill it with all ones
        if self.data["oc"] is None:
            self.data["oc"] = np.ones(self.n)

        if self.data["nd"] is None:
            self.data["nd"] = np.ones(self.n)

        self._check_to_numpy()

        # fill adds_oc, adds_oc_nd
        idx = np.where(self.data["oc"] == 1)[0]
        self.data["adds_oc"] = self.data["adds"][idx]

        idx = np.where(self.data["oc"] * self.data["nd"] == 1)
        self.data["adds_oc_nd"] = self.data["adds"][idx]

        # fill adds_th_score, adds_oc_th_score, adds_oc_nd_th_score
        self.data["adds_th_score"] = (self.data["adds"] <= self.data["adds_th"]).mean()
        # fill adds_auc, adds_oc_auc, adds_oc_nd_auc
        self.data["adds_auc"] = get_auc(self.data["adds"], self.data["adds_auc_th"])

        # oc
        if len(self.data["adds_oc"]) == 0:
            self.data["adds_oc_th_score"] = np.asarray([0.0])[0]
            self.data["adds_oc_auc"] = np.asarray([0.0])[0]
        else:
            self.data["adds_oc_th_score"] = (self.data["adds_oc"] <= self.data["adds_th"]).mean()
            self.data["adds_oc_auc"] = get_auc(self.data["adds_oc"], self.data["adds_auc_th"])

        # nd
        if len(self.data["adds_oc_nd"]) == 0:
            self.data["adds_oc_nd_th_score"] = np.asarray([0.0])[0]
            self.data["adds_oc_nd_auc"] = np.asarray([0.0])[0]
        else:
            self.data["adds_oc_nd_th_score"] = (self.data["adds_oc_nd"] <= self.data["adds_th"]).mean()
            self.data["adds_oc_nd_auc"] = get_auc(self.data["adds_oc_nd"], self.data["adds_auc_th"])

    def compute_oc(self):

        if self.data["oc"] is None or self.data["adds_oc"] is None:
            self.complete_eval_data()

        idx = np.where(self.data["oc"] == 1)[0]
        adds_oc = self.data["adds"][idx]
        rerr_oc = self.data["rerr"][idx]
        terr_oc = self.data["terr"][idx]

        OC = EvalData()
        OC.set_adds(adds_oc)
        OC.set_rerr(rerr_oc)
        OC.set_terr(terr_oc)
        OC.set_adds_th(self.data["adds_th"])
        OC.set_adds_auc_th(self.data["adds_auc_th"])
        OC.complete_eval_data()

        return OC

    def compute_ocnd(self):

        if self.data["oc"] is None or self.data["adds_oc"] is None:
            self.complete_eval_data()

        if self.data["nd"] is None or self.data["adds_oc_nd"] is None:
            self.complete_eval_data()

        idx = np.where(self.data["oc"] * self.data["nd"] == 1)
        adds_ocnd = self.data["adds"][idx]
        rerr_ocnd = self.data["rerr"][idx]
        terr_ocnd = self.data["terr"][idx]

        OCND = EvalData()
        OCND.set_adds(adds_ocnd)
        OCND.set_rerr(rerr_ocnd)
        OCND.set_terr(terr_ocnd)
        OCND.set_adds_th(self.data["adds_th"])
        OCND.set_adds_auc_th(self.data["adds_auc_th"])
        OCND.complete_eval_data()

        return OCND

    def _check_to_numpy(self):

        if isinstance(self.data["adds"], list):
            self.data["adds"] = np.asarray(self.data["adds"])

        if isinstance(self.data["rerr"], list):
            self.data["rerr"] = np.asarray(self.data["rerr"])

        if isinstance(self.data["terr"], list):
            self.data["terr"] = np.asarray(self.data["terr"])

        if isinstance(self.data["oc"], list):
            self.data["oc"] = np.asarray(self.data["oc"])

        if isinstance(self.data["nd"], list):
            self.data["nd"] = np.asarray(self.data["nd"])

    def print(self):
        """prints out the results"""

        raise NotImplementedError

    def save(self, filename):
        """saves object as a pickle file"""
        # breakpoint()
        with open(filename, 'wb') as f:
            pickle.dump(self.data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        """load object from pickle file"""

        with open(filename, 'rb') as f:
            data_dict = pickle.load(f)
        self.data = data_dict
        # breakpoint()


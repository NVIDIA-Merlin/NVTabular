import time
import datetime
from benchmark_parsers import BenchFastAI
from asvdb import BenchmarkResult


class CriteoBenchFastAI(BenchFastAI):
    def __init__(self, val=6, split=None):
        super().__init__("Criteo", val=val, split=split)

    def get_epoch(self, line):
        epoch, t_loss, v_loss, roc, aps, o_time = line.split()
        t_loss = self.bres_loss(epoch, float(t_loss))
        v_loss = self.bres_loss(epoch, float(v_loss), l_type="valid")
        roc = self.bres_roc_auc(epoch, float(roc))
        aps = self.bres_aps(epoch, float(aps))
        o_time = self.bres_time(epoch, o_time)
        return [t_loss, v_loss, roc, aps, o_time]

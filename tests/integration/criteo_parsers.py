from benchmark_parsers import Benchmark, BenchFastAI, create_bench_result
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

    
class CriteoBenchHugeCTR(Benchmark):
    
    def __init__(self):
        super().__init__("HugeCTR")
    
    def get_epochs(self, output):
        epochs = []
        for line in output:
            if "AUC" in line:
                epochs.append(self.get_epoch(line))
        return epochs
    

    def get_epoch(self, line):
        split_line = line.split(",")
        iteration = int(split_line[0].split(":")[-1])
        auc = float(split_line[-1][:-2])
        bres_auc = create_bench_result(f"{self.name}_auc", [("iteration", iteration)], auc, "percent")
        return [bres_itr, bres_auc]
import time
import datetime
from asvdb import BenchmarkResult


class Benchmark():
    """
    Main general benchmark parsing class
    """
    def __init__(self, target_id, val=1, split=None):
        self.name = f"{target_id}"
        self.val = val
        self.split = split
        
    def get_dl_thru(self, full_time, num_rows, epochs, throughput):
        return create_bench_result(f"{self.name}_dataloader",
                                   [("time", full_time),
                                    ("rows", num_rows),
                                    ("epochs", epochs)],
                                   throughput,
                                   "rows/second")        
    
    def bres_loss(self, epoch, loss, l_type="train"):
        return create_bench_result(f"{self.name}_{l_type}_loss",
                                   [("epoch", epoch)],
                                   loss,
                                   "percent")


    def bres_rmspe(self, epoch, rmspe):
        return create_bench_result(f"{self.name}_exp_rmspe",
                                   [("epoch", epoch)],
                                   rmspe,
                                   "percent")
    
    def bres_acc(self, epoch, acc):
        return create_bench_result(f"{self.name}_exp_rmspe",
                                   [("epoch", epoch)],
                                   acc,
                                   "percent")

    def bres_roc_auc(self, epoch, acc):
        return create_bench_result(f"{self.name}_exp_rmspe",
                                   [("epoch", epoch)],
                                   acc,
                                   "percent")
    
    
    def bres_time(self, epoch, r_time, time_format='%M:%S'):
        x = time.strptime(r_time.split(',')[0],time_format)
        r_time = datetime.timedelta(
                    hours=x.tm_hour,
                    minutes=x.tm_min,
                    seconds=x.tm_sec
                ).total_seconds()
        return create_bench_result(f"{self.name}_time",
                                   [("epoch", epoch)],
                                   r_time,
                                   "seconds")
    def bres_aps(self, epoch, aps):
        return create_bench_result(f"{self.name}_Avg_Prec",
                                   [("epoch", epoch)],
                                   aps,
                                   "percent")
    
    def get_dl_timing(self, output):
        timing_res = []
        for line in output:
            if line.startswith("run_time"):
                run_time, num_rows, epochs, dl_thru = line.split(" - ")
                run_time = float(run_time.split(": ")[1])
                num_rows = int(num_rows.split(": ")[1])
                epochs = int(epochs.split(": ")[1])
                dl_thru = float(dl_thru.split(": ")[1])
                bres = self.get_dl_thru(run_time, num_rows, epochs, dl_thru)
                timing_res.append(bres)
        return timing_res

    def get_epoch(self, line):
        raise NotImplementedError("Must Define logic for parsing metrics per epoch")
        
    def get_epochs(self, output):
        raise NotImplementedError("Must Define logic for parsing output")


        
### Sub classes

class BenchFastAI(Benchmark):
    def __init__(self, target_id, val=6, split=None):
        super().__init__(f"{target_id}_fastai", val=val, split=split)
    
    def get_epochs(self, output):
        epochs = []
        for line in output:
            split_line = line.split(self.split) if self.split else line.split()
            if len(split_line) == self.val and is_number(split_line[0]):
                # epoch line, detected based on if 1st character is a number
                post_evts = self.get_epoch(line)
                epochs.append(post_evts)
        return epochs
    

def is_number(str_to_num):
    try:
        val = int(str_to_num)
        return True
    except ValueError:
        return False
                
def send_results(db, bench_info, results_list):
    for results in results_list:
        if isinstance(results, list):
            for result in results:
                db.addResult(bench_info, result)
        else:
            db.addResult(bench_info, results)

def create_bench_result(name, arg_tuple_list, result, unit):
    return BenchmarkResult(funcName=name,
                           argNameValuePairs=arg_tuple_list,
                           unit=unit,
                           result=result)
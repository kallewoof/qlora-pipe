import os.path
import sys
from datetime import datetime, timedelta

import numpy as np
import torch


sys.path.insert(0, os.path.abspath('axolotl/src'))

from axolotl.utils.distributed import is_main_process, zero_first  # type: ignore # noqa

DTYPE_MAP = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}


# Simplified logger-like printer.
def log(msg):
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]}] [INFO] [qlora-pipe] {msg}')


def eta_str(eta):
    eta = int(eta)
    if eta > 3600:
        doc = str(datetime.now() + timedelta(seconds=eta)).rsplit(".", 1)[0]
        if eta < 86400:
            doc = doc[11:]
        return f'{eta // 3600}h{(eta % 3600) // 60}m {doc}'
    return f'{eta // 60}m{eta % 60}s' if eta > 60 else f'{eta}s'

def count_str(num):
    num = int(num)
    if num > 1000000000:
        return f"{num/1000000000:.2f}G"
    if num > 1000000:
        return f"{num/1000000:.2f}M"
    elif num > 1000:
        return f"{num/1000:.2f}k"
    return str(num)

class WeightedEstimator:
    """
    Estimates the time to complete a set of jobs of different weights.
    """
    def __init__(self, total_steps: int, total_weight: int, estimator_sz: 30):
        self.total_steps = total_steps
        self.total_weight = total_weight
        self.estimator_sz = estimator_sz
        self.rem_weight = total_weight
        self.rem_steps = total_steps
        self.times, self.weights = [], []
        self.polyfit = None

    def completed(self, time, weight):
        self.rem_weight -= weight
        self.rem_steps -= 1
        self.times.append(time)
        self.weights.append(weight)
        self.polyfit = np.polyfit(self.weights, self.times, 1)
        if len(self.times) > self.estimator_sz:
            self.times.pop(0)
            self.weights.pop(0)

    def eta(self):
        return self.polyfit[1] * (self.total_steps - self.finished_steps) + self.polyfit[0] * (self.total_weight - self.finished_weight)

    def __call__(self, added: float):
        return eta_str(added + self.eta())

def utfplot(eval_loss, eval_steps=100, unseen_steps=0, return_also=False):
    try:
        import plotille
    except ImportError:
        # Skipping plots
        return

    # Create the plot
    fig = plotille.Figure()
    fig.width = 60
    fig.height = 20
    fig.set_x_limits(min_=unseen_steps)
    fig.x_label = 'Step'
    fig.y_label = 'Evaluation Loss'

    for i in range(1, len(eval_loss)):
        eval_step = i * eval_steps + unseen_steps
        color = "red" if eval_loss[i] > eval_loss[i - 1] else "green"
        fig.plot([eval_step - eval_steps,eval_step], eval_loss[i-1:i+1], lc=color)

    # Print the plot
    f = fig.show()
    print(f)
    if return_also:
        return f

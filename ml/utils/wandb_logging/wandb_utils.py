"""Minimal no-op Weights & Biases compatibility layer for local training."""


def check_wandb_resume(opt):
    return None


class WandbLogger:
    def __init__(self, opt, name, run_id, data_dict, job_type="Training"):
        self.job_type = job_type
        self.wandb = None
        self.wandb_run = None
        self.data_dict = data_dict
        self.current_epoch = 0

    def log(self, payload):
        return None

    def end_epoch(self, best_result=False):
        return None

    def log_model(self, path, opt, epoch, fitness_score, best_model=False):
        return None

    def finish_run(self):
        return None

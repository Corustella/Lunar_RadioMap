"""Resumable patience on the official validation masked RMSE in dB."""

import math


class EarlyStopping:
    def __init__(self, patience=0, min_epochs=0, min_delta=0.0):
        if patience < 0 or min_epochs < 0 or not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError("Early stopping needs nonnegative patience, minimum epochs and finite min delta")
        self.config = {"patience": patience, "min_epochs": min_epochs, "min_delta_db": min_delta}
        self.best = float("inf")
        self.bad_epochs = 0
        self.stopped = False

    def update(self, metric, epoch):
        if not self.config["patience"]:
            return False
        if not math.isfinite(metric):
            raise ValueError("Early stopping received a nonfinite validation masked RMSE")
        if metric < self.best - self.config["min_delta_db"]:
            self.best = metric
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        self.stopped = (epoch + 1 >= self.config["min_epochs"]
                        and self.bad_epochs >= self.config["patience"])
        return self.stopped

    def state_dict(self):
        if not self.config["patience"]:
            return None
        return {"config": self.config.copy(), "best": self.best if math.isfinite(self.best) else None,
                "bad_epochs": self.bad_epochs, "stopped": self.stopped}

    def load_state_dict(self, state):
        expected = self.config if self.config["patience"] else None
        previous = state["config"] if state is not None else None
        if previous != expected:
            raise ValueError("Resume requires the same early-stopping policy")
        if state is not None:
            self.best = state["best"] if state["best"] is not None else float("inf")
            self.bad_epochs = state["bad_epochs"]
            self.stopped = state["stopped"]

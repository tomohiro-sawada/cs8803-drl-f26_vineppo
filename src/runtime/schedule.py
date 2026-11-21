"""Per-optimizer-step polynomial decay with 3% linear warmup."""

from math import ceil

from torch.optim.lr_scheduler import LambdaLR

POLYNOMIAL = "polynomial"


class PolynomialSchedule:
    def __init__(self, optimizers, total_steps):
        self.total_steps = total_steps
        self.warmup = ceil(0.03 * total_steps)
        assert 0 < self.warmup < total_steps, "Need optimizer steps after warmup"
        self.schedulers = [
            LambdaLR(optimizer, lambda step: self.factor(step))
            for optimizer in optimizers
        ]
        for optimizer, scheduler in zip(optimizers, self.schedulers, strict=True):
            optimizer.register_step_post_hook(
                lambda *_args, scheduler=scheduler: scheduler.step()
            )

    def factor(self, step):
        assert 0 <= step <= self.total_steps, "Optimizer exceeded scheduled steps"
        if step < self.warmup:
            return step / self.warmup
        return (self.total_steps - step) / (self.total_steps - self.warmup)

    def identity(self):
        return {
            "name": POLYNOMIAL,
            "power": 1,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup,
            "base_lrs": [scheduler.base_lrs for scheduler in self.schedulers],
        }

    def state_dict(self):
        return {
            "identity": self.identity(),
            "state": [scheduler.state_dict() for scheduler in self.schedulers],
        }

    def load_state_dict(self, state, expected_step):
        assert state["identity"] == self.identity(), "Learning-rate schedule mismatch"
        for scheduler, saved in zip(self.schedulers, state["state"], strict=True):
            assert saved["last_epoch"] == expected_step, "Schedule step mismatch"
            expected = [lr * self.factor(expected_step) for lr in scheduler.base_lrs]
            assert (
                expected
                == saved["_last_lr"]
                == [group["lr"] for group in scheduler.optimizer.param_groups]
            ), "Schedule learning rate mismatch"
            scheduler.load_state_dict(saved)

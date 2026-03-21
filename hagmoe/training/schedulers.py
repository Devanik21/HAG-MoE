import torch

class LambdaScheduler:
    def __init__(self, target_value: float, warmup_steps: int):
        self.target_value = target_value
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1

    def get_value(self) -> float:
        if self.current_step >= self.warmup_steps:
            return self.target_value
        return self.target_value * (self.current_step / self.warmup_steps)

class HAGMoEScheduler:
    def __init__(self, lambda_lb_g_target: float = 0.1, lambda_lb_e_target: float = 0.1,
                 lambda_div_target: float = 0.01, lambda_gamma_target: float = 0.01, warmup_steps: int = 5000):
        self.lb_g_scheduler = LambdaScheduler(lambda_lb_g_target, warmup_steps)
        self.lb_e_scheduler = LambdaScheduler(lambda_lb_e_target, warmup_steps)
        self.div_scheduler = LambdaScheduler(lambda_div_target, warmup_steps)
        self.gamma_scheduler = LambdaScheduler(lambda_gamma_target, warmup_steps)
        self.current_step = 0

    def step(self):
        self.lb_g_scheduler.step()
        self.lb_e_scheduler.step()
        self.div_scheduler.step()
        self.gamma_scheduler.step()
        self.current_step += 1

    def get_lambdas(self) -> dict:
        return {
            'lambda_lb_g': self.lb_g_scheduler.get_value(),
            'lambda_lb_e': self.lb_e_scheduler.get_value(),
            'lambda_div': self.div_scheduler.get_value(),
            'lambda_gamma': self.gamma_scheduler.get_value()
        }

def get_lr_scheduler(optimizer, warmup_steps, max_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793))).item()
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

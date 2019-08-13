import numpy as np

class Range():
    """Generate and narrow a range within specified boundaries."""

    def __init__(self, start, stop, n_steps, dtype=float):
        if stop < start: raise RuntimeError("'start' value of the range has a greater value then 'stop'")
        if n_steps < 3: raise ValueError("'n_steps' value is less then 3")

        self.start = start
        self.stop = stop
        self.n_steps = n_steps
        self.dtype = dtype
        self.init = True

    def generate(self):
        """Generate the range values."""

        if self.init == True:
            start = self.start
            stop = self.stop
            n = self.n_steps + 1
        else:
            step = self.step()
            start = self.start + step
            stop = self.stop - step
            n = self.n_steps - 1

        return np.linspace(start=start, stop=stop, num=n, dtype=self.dtype)
    
    def adjast(self, value):
        """Adjust the range to new narrow boundaries around the value."""

        step = self.step()
        self.start = max(value - step, self.start)
        self.stop = min(value + step, self.stop)
        self.init = False

    def step(self):
        return (self.stop - self.start) / self.n_steps

class LambdaRange(Range):
    """Generate and narrow a range of lambda regularization hyperparameter values."""

    def __init__(self, start=0.0001, stop=10, n_steps=3, *args, **kwargs):
        super(LambdaRange, self).__init__(start=start, stop=stop, n_steps=n_steps, *args, **kwargs)

    def adjast(self, hparam):
        if "lambda" not in hparam: raise ValueError("missing 'lambda' property of the 'hparam' dictionary")

        super().adjast(hparam["lambda"])

    def generate(self):
        return list(map(lambda v: {"lambda": v}, super().generate()))

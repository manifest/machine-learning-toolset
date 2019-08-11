import numpy as np

class Range():
    """Generate and narrow a range within specified boundaries."""

    def __init__(self, start, stop, steps, dtype=float):
        self.start = start
        self.stop = stop
        self.n_steps = steps
        self.dtype = dtype

        self._update_step()

    def generate(self):
        """Generate the range values."""

        return np.linspace(start=self.start, stop=self.stop, num=self.n_steps, dtype=self.dtype)
    
    def adjast(self, value):
        """Adjust the range to new narrow boundaries around the value."""

        self.start = max(value - self.step, self.start)
        self.stop = min(value + self.step, self.stop)

        self._update_step()
        
    def _update_step(self):
        if self.stop < self.start: raise RuntimeError("'start' value of the range has a greater value then 'stop'")

        self.step = (self.stop - self.start) / self.n_steps

class LambdaRange(Range):
    """Generate and narrow a range of lambda regularization hyperparameter values."""

    def __init__(self, start=0.0001, stop=10, steps=3, *args, **kwargs):
        super(LambdaRange, self).__init__(start=start, stop=stop, steps=steps, *args, **kwargs)

    def adjast(self, hparam):
        if "lambda" not in hparam: raise ValueError("missing 'lambda' property of the 'hparam' dictionary")

        super().adjast(hparam["lambda"])

    def generate(self):
        return list(map(lambda v: {"lambda": v}, super().generate()))

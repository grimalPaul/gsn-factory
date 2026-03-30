from typing import List


class RunningMean:
    def __init__(self, mean=0.0, n=0):
        self.mean = mean
        self.n = n

    def update(self, new_value):
        self.mean = self.running_mean(self.mean, self.n, new_value)
        self.n += 1

    def running_mean(self, mean, n, new_value):
        return (mean * n + new_value) / (n + 1)

    def get(self):
        return self.mean

    def reset(self):
        self.mean = 0.0
        self.n = 0


class RunningMeanDict:
    def __init__(self):
        self.means = {}

    def set_params(self, idx: int, mean: float, n: int):
        self.means[idx] = RunningMean(mean=mean, n=n)

    def update(self, new_value, idx: int):
        if idx not in self.means:
            self.means[idx] = RunningMean()
        self.means[idx].update(new_value)

    def get(self):
        return {k: v.get() for k, v in self.means.items()}


class RunningMeanList:
    def __init__(self, n: int):
        self.means = [RunningMean() for _ in range(n)]

    def set_params(self, idx: int, mean: float, n: int):
        self.means[idx] = RunningMean(mean=mean, n=n)

    def update_all(self, new_values: List[float]):
        for i, v in enumerate(new_values):
            if v is not None:
                self.means[i].update(v)

    def update_with_idx(self, new_value: float, idx: int):
        self.means[idx].update(new_value)

    def get(self):
        return [v.get() for v in self.means]

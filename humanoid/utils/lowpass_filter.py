import torch
import math

class LowPassFilter2ndOrder:
    def __init__(self, dt, cutoff_freq, env_num, device="cuda:0"):
        sample_freq = 1.0 / dt
        cutoff_freq = torch.tensor(cutoff_freq, device=device)
        assert sample_freq > 0.0 and torch.min(cutoff_freq) > 0.0 and torch.max(cutoff_freq) < sample_freq / 2.0

        self.dim_num = cutoff_freq.size(0)
        self.env_num = env_num
        self._sample_freq = sample_freq
        self._cutoff_freq = cutoff_freq

        self._delay_element_1 = torch.zeros((self.env_num, self.dim_num), device=device)
        self._delay_element_2 = torch.zeros((self.env_num, self.dim_num), device=device)

        fr = sample_freq / cutoff_freq
        ohm = torch.tan(math.pi / fr)
        c = 1.0 + 2.0 * math.cos(math.pi / 4.0) * ohm + ohm * ohm

        self._b0 = ohm * ohm / c
        self._b1 = 2.0 * self._b0
        self._b2 = self._b0

        self._a1 = 2.0 * (ohm * ohm - 1.0) / c
        self._a2 = (1.0 - 2.0 * math.cos(math.pi / 4.0) * ohm + ohm * ohm) / c

    def update(self, inputs):
        delay_element_0 = inputs - self._delay_element_1 * self._a1 - self._delay_element_2 * self._a2
        output = delay_element_0 * self._b0 + self._delay_element_1 * self._b1 + self._delay_element_2 * self._b2

        self._delay_element_2 = self._delay_element_1.clone()
        self._delay_element_1 = delay_element_0.clone()
        return output

    def reset(self, env_ids):
        self._delay_element_1[env_ids] *= 0
        self._delay_element_2[env_ids] *= 0

if __name__ == '__main__':
    fil = LowPassFilter2ndOrder(0.01, [1.0, 2.0, 3.0], 1024)

    inputs = torch.randn(1024, 3).to("cuda:0")
    outputs = fil.update(inputs)
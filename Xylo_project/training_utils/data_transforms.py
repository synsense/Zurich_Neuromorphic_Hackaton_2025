from dataclasses import dataclass
import numpy as np
from typing import  Union
from rockpool.devices.xylo.syns65302 import AFESimExternal

@dataclass
class SwapAxes:
    ax1: int = 0
    ax2: int = 1

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return np.swapaxes(data, self.ax1, self.ax2)


@dataclass
class AFESim3:
    fs: float = 16000
    spike_gen_mode: str = None
    fixed_threshold: Union[int, None] = None
    dn_EPS: int = 32
    rate_scale_factor: int = 63
    low_pass_averaging_window: float = 0.084

    """Sampling rate of the audio samples"""

    dt: float = 0.001024
    """dt of the network, the SNN core"""

    def __post_init__(self) -> None:
        if self.spike_gen_mode == "threshold":
            self.fixed_threshold_vec = [self.fixed_threshold for i in range(16)]
            self.dn_inits = {'spike_gen_mode':self.spike_gen_mode,'fixed_threshold_vec':self.fixed_threshold}    
        else:     
            self.fixed_threshold_vec = None
            self.dn_inits = {'spike_gen_mode':self.spike_gen_mode,'fixed_threshold_vec':self.fixed_threshold, 'dn_EPS': self.dn_EPS, 'rate_scale_factor': self.rate_scale_factor,
            'low_pass_averaging_window':self.low_pass_averaging_window}    

        self.afesim3 = AFESimExternal.from_specification(**self.dn_inits, dt = self.dt)

    def __call__(self,audio: np.ndarray) -> np.ndarray:
        if len(audio.shape) != 2 or audio.shape[0] != 1:
            raise ValueError("Audio should be in the form of (1,T)!")
         
        out,_,_ = self.afesim3((audio[0], self.fs))
        
        return out.T
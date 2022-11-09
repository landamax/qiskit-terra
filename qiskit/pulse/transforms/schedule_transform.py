import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from qiskit.providers.fake_provider import FakeAthens 
from qiskit import pulse
from qiskit.visualization.pulse_v2.events import ChannelEvents
from qiskit.visualization.pulse_v2.types import PulseInstruction
from qiskit.pulse.transforms.base_transforms import target_qobj_transform
from qiskit.pulse import Schedule, Play, Gaussian, Constant, DriveChannel, Delay, ParametricPulse, SymbolicPulse

def get_time_step_array(waveform: PulseInstruction) -> Tuple[int, int, np.array, np.array]:
    t0 = waveform.t0
    if isinstance(waveform.inst, Play):
        tf = t0 + waveform.inst.pulse.duration
    if isinstance(waveform.inst, Delay):
        tf = t0 + waveform.inst.duration
    t_arr = np.arange(t0, tf)
    waveform_samples = np.ones(t_arr.size, dtype=complex)
    return t0, tf, t_arr, waveform_samples


def combine_transmit_schedule_to_single_waveform(sched: Schedule, apply_carrier_wave: bool) -> pulse.Waveform:
    """ combine all instructions in all the channels of sched to a single 
    Waveform object """
    #TODO: add cw according to the qubit freq from backend.defaults().qubit_freq_est
    
    sched = target_qobj_transform(sched)
    total_samples = np.ones(sched.duration, dtype=complex)
    for channel in sched.channels:
        channel_events = ChannelEvents.load_program(sched, channel)
        channel_samples = np.ones(sched.duration, dtype=complex)
        for waveform in channel_events.get_waveforms():
            if not isinstance(waveform.inst, Play) and \
                    not isinstance(waveform.inst, Delay):
                continue
            if isinstance(waveform.inst, Play):
                print(waveform.inst.pulse)
                freq = waveform.frame.freq
                phase = waveform.frame.phase
                t0, tf, t_arr, waveform_samples = get_time_step_array(waveform)
                if isinstance(waveform.inst.pulse, ParametricPulse) or \
                        isinstance(waveform.inst.pulse, SymbolicPulse):
                    waveform_samples = waveform.inst.pulse.get_waveform().samples
                else:
                    waveform_samples = waveform.inst.pulse.samples
                waveform_samples *= np.exp(1j * 2*np.pi * freq * t_arr) 
                waveform_samples *= np.exp(1j * phase)
            if isinstance(waveform.inst, Delay):
                t0, tf, _, waveform_samples = get_time_step_array(waveform)
            if apply_carrier_wave:
                waveform_samples *= np.exp(1j * 2*np.pi * backend.defaults().qubit_freq_est * t_arr)
            channel_samples[t0:tf] = waveform_samples
        total_samples *= channel_samples
    return pulse.Waveform(total_samples)

backend = FakeAthens()
print("fakeAthens drive channel timestep: "+str(backend.configuration().dt))
# parameters
base_duration = 1024
base_amp = 1
sigma = 256

# create pulses and channels
gaussian = Gaussian(base_duration, base_amp, sigma)
const = Constant(int(0.5*base_duration), 0.5*base_amp)
short_gaussian = Gaussian(int(0.25*base_duration), base_amp, 0.25*sigma)
channel1 = DriveChannel(0)
channel2 = DriveChannel(1)

# create a schedule with multiple channels
with pulse.build('aer_simulator') as sched:
    backend = FakeValencia()
    #pulse.set_frequency(1e9, channel1)
    pulse.play(gaussian, channel1)
    pulse.delay(int(0.1*base_duration), channel2)
    pulse.set_phase(np.pi, channel2)
    pulse.play(const, channel2)
    pulse.play(pulse.library.sawtooth(int(0.1*base_duration), base_amp), channel1)
    #pulse.play(short_gaussian, channel2)
    print("qubit freq is: "+str(backend.defaults().qubit_freq_est))
sched.draw()

total_waveform = combine_transmit_schedule_to_single_waveform(sched)

plt.figure()
plt.title("combined waveform")
plt.plot(total_waveform.samples)
plt.show()

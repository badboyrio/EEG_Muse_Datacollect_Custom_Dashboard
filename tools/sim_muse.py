"""Fake Muse LSL stream for testing without the headband.

Publishes 4 EEG channels + AUX at 256 Hz: smooth noise, 10 Hz alpha and
blink-like dips on TP9 every 2-4 s.

    python tools/sim_muse.py               # all channels look GOOD
    python tools/sim_muse.py --dead-tp10   # TP10 flat, as with no skin contact

Then run record_session.py (or the dashboard) as usual instead of muselsl stream.
"""
import sys, time, numpy as np
from pylsl import StreamInfo, StreamOutlet, local_clock
dead_tp10 = "--dead-tp10" in sys.argv
FS = 256
info = StreamInfo("Muse-SIM", "EEG", 5, FS, "float32", "muse-sim")
chs = info.desc().append_child("channels")
for n in ["TP9", "AF7", "AF8", "TP10", "Right AUX"]:
    chs.append_child("channel").append_child_value("label", n)
out = StreamOutlet(info, 32)
rng = np.random.default_rng(0); t = 0.0; state = np.zeros(5)
next_blink = 2.0
while True:
    n = 16; tt = t + np.arange(n) / FS
    state = 0.97 * state + rng.normal(0, 3, 5)  # smooth-ish noise
    x = np.tile(state, (n, 1)) + rng.normal(0, 6, (n, 5))
    x[:, :4] += 8 * np.sin(2 * np.pi * 10 * tt)[:, None]
    for i, ti in enumerate(tt):
        if abs(ti - next_blink) < 0.05: x[i, 0] -= 90 * np.exp(-((ti - next_blink) / 0.03) ** 2)
    if tt[-1] > next_blink + 0.1: next_blink += rng.uniform(2, 4)
    if dead_tp10: x[:, 3] = 0.2 * rng.normal(size=n)
    out.push_chunk(x.tolist())
    t += n / FS; time.sleep(n / FS)

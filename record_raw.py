"""
Continuous raw EEG recording from the Muse LSL stream (no task).
Writes every chunk to data/raw/<timestamp>/eeg.csv as it arrives, so nothing
is lost if the process is stopped. Stop with Ctrl+C.
"""
import argparse
import csv
import os
import signal
import time

from pylsl import StreamInlet, proc_clocksync, proc_dejitter, resolve_byprop


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(here, "data", "raw", time.strftime("%Y%m%d_%H%M%S")))
    args = ap.parse_args()

    streams = resolve_byprop("type", "EEG", timeout=15)
    if not streams:
        raise SystemExit("No EEG stream found. Start the Muse stream first.")
    inlet = StreamInlet(streams[0], max_buflen=360, processing_flags=proc_clocksync | proc_dejitter)
    info = inlet.info()
    ch = info.desc().child("channels").child("channel")
    names = []
    for _ in range(info.channel_count()):
        names.append(ch.child_value("label"))
        ch = ch.next_sibling()

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "eeg.csv")
    running = [True]
    signal.signal(signal.SIGTERM, lambda *_: running.__setitem__(0, False))

    n, t0 = 0, None
    print(f"Recording {names} -> {path}", flush=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lsl_time", "t_rel"] + names)
        try:
            while running[0]:
                data, ts = inlet.pull_chunk(timeout=1.0)
                if not ts:
                    continue
                if t0 is None:
                    t0 = ts[0]
                for t, row in zip(ts, data):
                    w.writerow([f"{t:.6f}", f"{t - t0:.6f}"] + [f"{v:.4f}" for v in row])
                f.flush()
                n += len(ts)
                if n % (256 * 30) < len(ts):
                    print(f"{n} samples ({n / 256 / 60:.1f} min)", flush=True)
        except KeyboardInterrupt:
            pass
    print(f"Saved {n} samples ({n / 256 / 60:.1f} min) to {path}")


if __name__ == "__main__":
    main()

"""
Muse EEG recording session: mental workload (arithmetic vs rest).

Usage (with `muselsl stream` already running in another terminal):
    python3 record_session.py --sub 01 --ses 01

Controls:
    SPACE = start session (from signal-quality screen)
    ESC   = abort and save what was recorded

Screen:
    Top bar shows the current phase, its countdown, block progress
    and total session time remaining.

Output (data/sub-XX/ses-YY/):
    eeg.csv     lsl_time, t_rel, channels..., label
    events.csv  onset, duration, trial_type, block, start_number
"""
import argparse
import csv
import os
import platform
import random
import threading
import time
import tkinter as tk

import numpy as np
from pylsl import StreamInlet, local_clock, proc_clocksync, proc_dejitter, resolve_byprop

# ---------------- Protocol settings ----------------
N_BLOCKS_PER_CLASS = 10
BLOCK_SEC = 40
BREAK_SEC = 8
BASELINE_CLOSED_SEC = 120
BASELINE_OPEN_SEC = 120
END_BASELINE_SEC = 60
READY_SEC = 5
FS_EXPECTED = 256
# ---------------------------------------------------

IS_MAC = platform.system() == "Darwin"
MONO_FONT = "Menlo" if IS_MAC else "Consolas"
TEXT_FONT = "Helvetica" if IS_MAC else "Arial"

PHASE_NAMES = {
    "ready": "Get ready",
    "baseline_eyes_closed": "Baseline - eyes closed",
    "baseline_eyes_open": "Baseline - eyes open",
    "rest": "REST",
    "arithmetic": "ARITHMETIC",
    "break": "Break",
    "end_baseline_eyes_open": "Final baseline - eyes open",
}


def fmt(sec):
    sec = max(0, int(round(sec)))
    return f"{sec // 60:02d}:{sec % 60:02d}"


class Recorder(threading.Thread):
    def __init__(self, inlet):
        super().__init__(daemon=True)
        self.inlet = inlet
        self.ts, self.data = [], []
        self.lock = threading.Lock()
        self.running = True

    def run(self):
        while self.running:
            chunk, stamps = self.inlet.pull_chunk(timeout=1.0, max_samples=512)
            if stamps:
                with self.lock:
                    self.data.extend(chunk)
                    self.ts.extend(stamps)

    def recent(self, n):
        with self.lock:
            return np.array(self.data[-n:]) if self.data else np.empty((0, 0))

    def snapshot(self):
        with self.lock:
            return np.array(self.ts), np.array(self.data)


def channel_names(info):
    names = []
    ch = info.desc().child("channels").child("channel")
    for _ in range(info.channel_count()):
        label = ch.child_value("label")
        names.append(label if label else f"ch{len(names)+1}")
        ch = ch.next_sibling()
    if all(n.startswith("ch") for n in names):
        names = ["TP9", "AF7", "AF8", "TP10", "AUX"][: info.channel_count()]
    return names


def build_timeline():
    tl = [("ready", READY_SEC, "Get ready...\n\nSit still, relax your jaw.", None)]
    tl.append(("baseline_eyes_closed", BASELINE_CLOSED_SEC,
               "Close your eyes and relax.\n\nOpen them when you hear the beep.", None))
    tl.append(("baseline_eyes_open", BASELINE_OPEN_SEC, "+", None))
    for b in range(N_BLOCKS_PER_CLASS):
        pair = ["rest", "arithmetic"]
        random.shuffle(pair)
        for cls in pair:
            if cls == "rest":
                tl.append(("rest", BLOCK_SEC, "+", b + 1))
            else:
                tl.append(("arithmetic", BLOCK_SEC, None, b + 1))
            tl.append(("break", BREAK_SEC, "Break\n\nblink, swallow, relax", b + 1))
    tl.append(("end_baseline_eyes_open", END_BASELINE_SEC, "+", None))
    return tl


class App:
    def __init__(self, root, rec, names, outdir):
        self.root, self.rec, self.names, self.outdir = root, rec, names, outdir
        self.timeline = build_timeline()
        self.total_sec = sum(p[1] for p in self.timeline)
        self.events = []
        self.idx = -1
        self.started = False
        self.saved = False
        self.phase_end = None
        self.trial_count = 0

        root.configure(bg="black")
        root.attributes("-fullscreen", True)
        root.bind("<space>", self.start)
        root.bind("<Escape>", self.abort)
        root.focus_force()

        # top bar: countdown / progress
        self.timer = tk.Label(root, text="", fg="#9a9a9a", bg="black",
                              font=(MONO_FONT, 24), justify="center")
        self.timer.pack(side="top", pady=(30, 0))
        # main stimulus
        self.label = tk.Label(root, text="", fg="white", bg="black",
                              font=(TEXT_FONT, 48), justify="center")
        self.label.pack(expand=True)
        # quality readout (only before start)
        self.quality = tk.Label(root, text="", fg="white", bg="black",
                                font=(MONO_FONT, 22), justify="left")
        self.quality.pack(pady=40)

        self.label.config(text="Signal quality check\n\nAdjust headband until all 4 are GOOD,\nthen press SPACE")
        self.timer.config(text=f"Session length: {fmt(self.total_sec)}")
        self.update_quality()

    # ---- signal quality screen ----
    def update_quality(self):
        if self.started:
            self.quality.config(text="")
            return
        x = self.rec.recent(2 * FS_EXPECTED)
        lines = []
        if x.size and x.shape[0] > FS_EXPECTED:
            for i, name in enumerate(self.names[:4]):
                s = float(np.std(x[:, i]))
                if s < 1:
                    status = "FLAT (no contact)"
                elif s < 30:
                    status = "GOOD"
                elif s < 80:
                    status = "OK - adjust"
                else:
                    status = "BAD - noisy/contact"
                lines.append(f"{name:>5}: {s:7.1f} uV  {status}")
        else:
            lines.append("Waiting for data...")
        self.quality.config(text="\n".join(lines))
        self.root.after(500, self.update_quality)

    # ---- countdown ----
    def tick(self):
        if self.saved or self.phase_end is None or self.idx >= len(self.timeline):
            return
        now = local_clock()
        kind, _, _, block = self.timeline[self.idx]
        phase_left = self.phase_end - now
        later = sum(p[1] for p in self.timeline[self.idx + 1:])
        session_left = phase_left + later

        parts = [f"{PHASE_NAMES.get(kind, kind)}  {fmt(phase_left)}"]
        if block:
            parts.append(f"Block {block}/{N_BLOCKS_PER_CLASS}  Trial {self.trial_count}/{2 * N_BLOCKS_PER_CLASS}")
        parts.append(f"Session left {fmt(session_left)}")
        self.timer.config(text="   |   ".join(parts))

        # during breaks, show a big countdown in the centre too
        if kind == "break":
            self.label.config(text=f"Break\n\nblink, swallow, relax\n\nnext in {int(max(0, phase_left)) + 1}")
        self.root.after(200, self.tick)

    # ---- session flow ----
    def start(self, _=None):
        if self.started:
            return
        self.started = True
        self.next_phase()
        self.tick()

    def next_phase(self):
        now = local_clock()
        if self.events:
            self.events[-1]["duration"] = now - self.events[-1]["onset_abs"]
        self.idx += 1
        if self.idx >= len(self.timeline):
            self.timer.config(text="Session complete")
            self.label.config(text="Done. Thank you!\n\nSaving...")
            self.root.after(1000, self.finish)
            return

        kind, dur, text, block = self.timeline[self.idx]
        start_number = ""
        if kind in ("rest", "arithmetic"):
            self.trial_count += 1
        if kind == "arithmetic":
            start_number = random.randint(500, 999)
            text = f"{start_number}\n\nSubtract 7 repeatedly, silently"
        if kind == "baseline_eyes_open" and self.timeline[self.idx - 1][0] == "baseline_eyes_closed":
            self.root.bell()  # signal end of eyes-closed baseline
        self.label.config(text=text)
        self.phase_end = now + dur
        self.events.append({"onset_abs": now, "duration": dur, "trial_type": kind,
                            "block": block or "", "start_number": start_number})
        self.root.after(int(dur * 1000), self.next_phase)

    def abort(self, _=None):
        if self.events:
            self.events[-1]["duration"] = local_clock() - self.events[-1]["onset_abs"]
            self.events[-1]["trial_type"] += "_aborted"
        self.finish()

    def finish(self):
        if self.saved:
            return
        self.saved = True
        time.sleep(0.5)  # drain last samples
        self.rec.running = False
        save(self.rec, self.names, self.events, self.outdir)
        self.root.destroy()


def save(rec, names, events, outdir):
    ts, data = rec.snapshot()
    if ts.size == 0:
        print("No EEG data recorded.")
        return
    os.makedirs(outdir, exist_ok=True)
    t0 = ts[0]

    labels = np.array(["none"] * len(ts), dtype=object)
    for ev in events:
        a = np.searchsorted(ts, ev["onset_abs"])
        b = np.searchsorted(ts, ev["onset_abs"] + ev["duration"])
        labels[a:b] = ev["trial_type"]

    eeg_path = os.path.join(outdir, "eeg.csv")
    with open(eeg_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lsl_time", "t_rel"] + names + ["label"])
        for t, row, lab in zip(ts, data, labels):
            w.writerow([f"{t:.6f}", f"{t - t0:.6f}"] + [f"{v:.4f}" for v in row] + [lab])

    ev_path = os.path.join(outdir, "events.csv")
    with open(ev_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["onset", "duration", "trial_type", "block", "start_number"])
        for ev in events:
            w.writerow([f"{ev['onset_abs'] - t0:.4f}", f"{ev['duration']:.4f}",
                        ev["trial_type"], ev["block"], ev["start_number"]])

    fs = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 else 0
    print(f"Saved {len(ts)} samples ({(ts[-1]-t0)/60:.1f} min, ~{fs:.1f} Hz)")
    print(f"  {eeg_path}\n  {ev_path}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", required=True, help="subject ID, e.g. 01")
    ap.add_argument("--ses", required=True, help="session ID, e.g. 01")
    ap.add_argument("--out", default=os.path.join(here, "data"))
    args = ap.parse_args()

    outdir = os.path.join(args.out, f"sub-{args.sub}", f"ses-{args.ses}")
    if os.path.exists(os.path.join(outdir, "eeg.csv")):
        raise SystemExit(f"{outdir} already has a recording. Use a new session number.")

    print("Looking for EEG stream (is the stream window running?)...")
    streams = resolve_byprop("type", "EEG", timeout=15)
    if not streams:
        raise SystemExit("No EEG stream found. Start 1_Start_Stream first.")
    inlet = StreamInlet(streams[0], max_buflen=360,
                        processing_flags=proc_clocksync | proc_dejitter)
    names = channel_names(inlet.info())
    print(f"Connected: {streams[0].name()}  channels={names}")

    rec = Recorder(inlet)
    rec.start()

    root = tk.Tk()
    App(root, rec, names, outdir)
    root.mainloop()


if __name__ == "__main__":
    main()

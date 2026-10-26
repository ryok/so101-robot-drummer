"""打音検出の確認: マイクの音量を0.1秒ごとに表示し、strike_conductor と同じ閾値で打音を数える。
使い方: python scripts/mic_check.py [--mic 3] [--secs 10]
  実行中に、スティックを手で持ってドラムとボウルを数回ずつ軽く叩く（ロボットは使わない）。
  「打音」が叩いた回数だけ出て、静かなときに出なければ、その --mic / --mic-k で使える。
"""
import argparse, time

import numpy as np
import sounddevice as sd

from strike_conductor import Mic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mic", type=int, default=3)
    ap.add_argument("--mic-k", type=float, default=5.0)
    ap.add_argument("--mic-min", type=float, default=0.05)
    ap.add_argument("--secs", type=float, default=10.0)
    a = ap.parse_args()
    print(f"マイク {a.mic}: {sd.query_devices(a.mic)['name']}  （最初の約1秒は静かに）")
    m = Mic(a.mic)
    th = m.set_threshold(a.mic_k, a.mic_min)
    print(f"静かなときの最大振幅 {m.floor:.4f} → 閾値 {th:.4f}。叩いてください")
    t_start = time.perf_counter()
    t_end, last = t_start + a.secs, t_start
    while time.perf_counter() < t_end:
        time.sleep(0.1)
        now = time.perf_counter()
        peak = max((np.abs(x).max() for adc, x in list(m.buf) if adc + m.off >= last), default=0.0)
        on = m.onset(last, now)
        print(f"{now - t_end + a.secs:5.1f}s  {'#' * min(60, int(peak / th * 10)):<60} {peak:.3f}"
              + ("  ← 打音" if on is not None else ""))
        last = now
    n = len(m.onsets(t_start, t_end))
    m.close()
    print(f"打音 {n} 回（叩いた回数と合っているか確認。0.15秒以内の続けての検出は1回と数える）")


if __name__ == "__main__":
    main()

"""叩きデータの前後の「待ち」を切り落として、新しいデータセットに書き出す。
録画中の人の反応時間（Recording 表示から振り下ろすまで）は毎回ばらつき、静止中の映像からは
「いつ動くか」が読めないので、ポリシーが動き出しを学べない。動き出しの直前から始まるように揃える。
使い方:
  python scripts/trim_strikes.py <元データのroot> local/so101-strike-v3-trim --dry   # 判定だけ表示
  python scripts/trim_strikes.py <元データのroot> local/so101-strike-v3-trim         # 書き出し
  --append <既存のtrim root>: 既存の切り出し済みデータセットに追記（ボウル分を足すとき）
  --from-ep N: 元データの ep N 以降だけ処理（--resume で録り足した分だけ追記するとき）
  --task-by-pan "Strike the drum,Strike the bowl": 叩いた方向（pan −=左 / +=右）でタスク名を付け直す。
      ドラムとボウルを1回の録画で交互に録るとき用（構えの姿勢が楽器ごとに偏らないようにするため）
動き出し: 叩いた瞬間（肘が最も下がったフレーム）から遡り、0.3秒じっと構えていた区間の終わり。
          冒頭の構え直しや、畳んだ姿勢から広げる動きは切り捨てる。
自動で除外: 叩いていない回（肘の振れ < --min-swing）、構えずに畳んだ姿勢から叩いた回、
          構えに戻りきらずに終わった回（最終フレームが構えから 8° 超ずれ）。
"""
import argparse, glob

import numpy as np
import pandas as pd

from lerobot.datasets.lerobot_dataset import LeRobotDataset

META = {"timestamp", "frame_index", "episode_index", "index", "task_index", "task"}


def plan(root, pre, post, min_swing, folded_elbow, fps, back_tol=8.0, still_s=0.6):
    """各エピソードの (開始, 終了) フレームと、除外理由を返す。動き出しは leader の命令(action)で判定。"""
    df = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{root}/data/**/*.parquet", recursive=True)])
    out = []
    for e, g in df.groupby("episode_index"):
        g = g.sort_values("frame_index")
        A = np.stack(g["action"].values)
        S = np.stack(g["observation.state"].values)
        # 叩いた瞬間 = 肘が最も下がった（+方向）フレーム。そこから遡って「still_s 秒じっとしていた区間」の
        # 終わりを動き出しとする。冒頭のもぞもぞ（構え直し）や、畳んだ姿勢から広げる動きはここで切り捨てられる
        step = np.r_[0, np.abs(np.diff(A[:, :5], axis=0)).max(1)]
        # 叩いた瞬間 = 直前1秒の最小値からの肘の「下がり幅」が最大のフレーム。畳んだ姿勢から広げる動きは
        # 肘が上がる(−方向)ので、ここでは拾わない
        el = S[:, 2]
        rise = np.array([el[i] - el[max(0, i - fps):i + 1].min() for i in range(len(el))])
        hit = int(np.argmax(rise))
        if rise[hit] < min_swing:
            out.append((e, None, None, None, f"叩いていない（肘の振れ{rise[hit]:.0f}°）"))
            continue
        w = round(still_s * fps)
        on = None
        for i in range(hit, round(0.2 * fps) - 1, -1):   # 窓内の振れ幅 2° 未満 = じっと構えている（手の揺れは許す）
            win = A[max(0, i - w + 1):i + 1, :5]           # 冒頭は窓が短くてもよい（すぐ叩いた回）
            if (win.max(0) - win.min(0)).max() < 2.0:
                on = i
                break
        if on is None:
            out.append((e, None, None, None, "叩く前にじっと構えた区間がない"))
            continue
        if S[on, 2] > folded_elbow:
            out.append((e, None, None, None, f"畳んだ姿勢から叩いている（肘{S[on, 2]:.0f}°）"))
            continue
        end_dev = np.abs(S[-1, :5] - S[on, :5]).max()
        if end_dev > back_tol:
            out.append((e, None, None, None, f"構えに戻りきらずに終了（最後に{end_dev:.0f}°ずれ）"))
            continue
        last_move = int(np.where(step > 0.5)[0][-1])
        s = max(0, on - round(pre * fps))
        t = min(len(A), last_move + 1 + round(post * fps))
        out.append((e, s, t, float(S[hit, 0] - S[on, 0]), ""))   # pan: −=左 / +=右
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src_root")
    ap.add_argument("dst_repo_id")
    ap.add_argument("--dst-root", default=None, help="既定は ~/.cache/huggingface/lerobot/<dst_repo_id>")
    ap.add_argument("--append", default=None, help="既存の切り出し済みデータセットのroot（追記）")
    ap.add_argument("--pre", type=float, default=0.15, help="動き出しの何秒前から残すか")
    ap.add_argument("--post", type=float, default=0.3, help="最後の動きの何秒後まで残すか")
    ap.add_argument("--min-swing", type=float, default=8.0)
    ap.add_argument("--folded-elbow", type=float, default=80.0)
    ap.add_argument("--back-tol", type=float, default=8.0,
                    help="叩いた後、構えからこの角度[deg]以内に戻っていれば採用（全関節の最大ずれ）")
    ap.add_argument("--from-ep", type=int, default=0, help="元データのこの番号以降だけ処理（--append で追加分だけ足すとき）")
    ap.add_argument("--task-by-pan", default=None,
                    help='叩いた方向でタスク名を付け直す "左のタスク,右のタスク"（楽器を混ぜて録った回用）')
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    src = LeRobotDataset("src", root=a.src_root)
    fps = src.fps
    rows = [r for r in plan(a.src_root, a.pre, a.post, a.min_swing, a.folded_elbow, fps, a.back_tol)
            if r[0] >= a.from_ep]
    keep = [r for r in rows if r[1] is not None]
    left_task, right_task = (a.task_by_pan.split(",") if a.task_by_pan else (None, None))
    task_of = lambda pan: (left_task if pan < 0 else right_task) if a.task_by_pan else None
    for e, s, t, pan, why in rows:
        print(f"ep{e}: " + (f"除外 — {why}" if s is None else
                            f"残す {s / fps:.2f}s〜{t / fps:.2f}s（{(t - s) / fps:.2f}s, {t - s}frame） pan{pan:+.0f}°"
                            + (f" → {task_of(pan)}" if a.task_by_pan else "")))
    print(f"→ {len(keep)}/{len(rows)}本を残す")
    if a.dry or not keep:
        return

    if a.append:
        dst = LeRobotDataset.resume(a.dst_repo_id, root=a.append)
    else:
        feats = {k: v for k, v in src.meta.features.items() if k not in META}
        dst = LeRobotDataset.create(a.dst_repo_id, fps, feats, root=a.dst_root,
                                    robot_type=src.meta.robot_type, use_videos=True)
    starts = src.meta.episodes["dataset_from_index"]
    for e, s, t, pan, _ in keep:
        base = int(starts[e])
        for i in range(base + s, base + t):
            it = src[i]
            frame = {"task": task_of(pan) or it["task"]}
            for k, v in it.items():
                if k in META:
                    continue
                if k.startswith("observation.images."):
                    v = (v.permute(1, 2, 0).clamp(0, 1) * 255).round().to(dtype=__import__("torch").uint8).numpy()
                else:
                    v = v.numpy()
                frame[k] = v
            dst.add_frame(frame)
        dst.save_episode()
        print(f"  書き出し ep{e} → 新ep{dst.meta.total_episodes - 1}")
    dst.finalize()
    print("出力:", dst.root, "合計", dst.meta.total_episodes, "本")


if __name__ == "__main__":
    main()

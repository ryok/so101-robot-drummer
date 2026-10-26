"""収集後QC: 各エピソードの先頭frontフレームをタイル化し、長さ異常を洗い出す。
使い方: python scripts/qc_dataset.py <dataset_root> [video_key]
- 画角ドリフト検査: タイル画像 outputs/qc_front_tiles.jpg を目視
- 長さ: 60s上限(≈1800frame)近くのエピソードをフラグ（だらだら録画/未完了の疑い）
"""
import sys, glob, math
import pandas as pd, numpy as np, cv2

def main():
    root = sys.argv[1].rstrip("/")
    key = sys.argv[2] if len(sys.argv) > 2 else "observation.images.front"
    import json
    info = json.load(open(f"{root}/meta/info.json"))
    fps = info["fps"]; cap_frames = 60 * fps
    vt = info["video_path"]  # videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4

    dfs = [pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/meta/episodes/**/*.parquet", recursive=True))]
    df = pd.concat(dfs, ignore_index=True).sort_values("episode_index").reset_index(drop=True)

    print(f"total_episodes={len(df)}  fps={fps}  60s上限≈{cap_frames}frame")
    print(f"{'ep':>3} {'len':>5} {'秒':>5}  {'先頭輝度':>7}  flag")
    frames, brights = [], []
    for _, r in df.iterrows():
        ep = int(r["episode_index"]); length = int(r["length"])
        ci = int(r[f"videos/{key}/chunk_index"]); fi = int(r[f"videos/{key}/file_index"])
        ts = float(r[f"videos/{key}/from_timestamp"])
        vp = f"{root}/" + vt.format(video_key=key, chunk_index=ci, file_index=fi)
        cap = cv2.VideoCapture(vp)
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
        ok, fr = cap.read(); cap.release()
        if not ok or fr is None:
            print(f"{ep:>3} {length:>5}  先頭フレーム取得失敗 ({vp})"); frames.append(None); brights.append(0); continue
        b = float(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).mean())
        flag = "⚠長さ上限近い" if length >= cap_frames - 30 else ""
        print(f"{ep:>3} {length:>5} {length/fps:>5.1f}  {b:>7.1f}  {flag}")
        cv2.putText(fr, f"ep{ep} len{length} b{b:.0f}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        frames.append(fr); brights.append(b)

    valid = [f for f in frames if f is not None]
    if valid:
        h, w = valid[0].shape[:2]
        cols = 5; rows = math.ceil(len(frames)/cols)
        tile = np.zeros((rows*h, cols*w, 3), np.uint8)
        for i, f in enumerate(frames):
            if f is None: continue
            rr, cc = divmod(i, cols)
            tile[rr*h:(rr+1)*h, cc*w:(cc+1)*w] = cv2.resize(f, (w, h))
        out = "outputs/qc_front_tiles.jpg"
        import os; os.makedirs("outputs", exist_ok=True)
        cv2.imwrite(out, tile)
        bs = np.array(brights)
        print(f"\nタイル保存: {out}")
        print(f"先頭輝度: min={bs.min():.0f} max={bs.max():.0f} 平均={bs.mean():.0f} 標準偏差={bs.std():.1f}"
              f"  {'（ばらつき大→画角/照明ドリフト疑い）' if bs.std() > 8 else '（安定）'}")

if __name__ == "__main__":
    main()

"""スマホのテイクで、ロボットの台（と腕が動く範囲）より外側にモザイクをかけた映像を作る。

手持ちで撮ったのでカメラが動く（最大で約120px、ズームで約20%）。そこで台の天板の特徴点を
1コマずつ追い（optical flow → 相似変換）、最初のコマで決めた「残す範囲」をそのコマに合わせて動かす。
コマ間の追跡だけだと誤差が積もるので、毎コマ最初のコマにも合わせ直す。

使い方: python video/mask_takes.py [--src ~/Downloads] [--out outputs/video/masked] [--only IMG_0363]
出力は映像だけ（音は build_video.py が元の MOV から取る）。縦長のテイクは回転を適用した 1080×1920 で書く。
"""
import argparse, os, subprocess
import cv2
import numpy as np

FPS = 30
TAKES = ("IMG_0344", "IMG_0363", "IMG_0365")
# 最初のコマ（t=0）の座標。KEEP = 残す範囲（台の天板 ∪ 腕が動く範囲）、TABLE = 追跡に使う天板。
KEEP = {
    # 天板の四隅を画面の外まで延ばし、奥の縁の上に腕が動く範囲（全コマの最大値で確かめた）を足す
    "IMG_0344": [(140, 118), (500, 100), (500, -400), (1540, -400), (1550, 140), (1665, 145), (1890, 1600), (70, 1600)],
    "IMG_0363": [(40, 650), (95, 585), (240, 585), (240, 360), (300, 290), (370, 240), (450, 222), (560, 215), (680, 222),
                 (760, 238), (795, 300), (785, 385), (860, 392), (910, 420), (920, 585), (1090, 585), (1100, 1195), (0, 1185),
                 (-20, 780)],
    # 腕の上の左右は奥のポスターと箱の文字なので、腕が動く範囲に沿わせる
    "IMG_0365": [(-600, 2600), (-600, 655), (0, 655), (0, 450), (140, 420), (200, 330), (260, 270), (330, 230), (450, 190),
                 (560, 130), (730, 130), (730, 240), (800, 250), (860, 340), (900, 420), (1000, 490), (1080, 560), (1080, 640),
                 (1700, 640), (1700, 2600)],
}
TABLE = {
    "IMG_0344": [(150, 130), (1650, 150), (1790, 1075), (105, 1075)],
    "IMG_0363": [(95, 590), (1075, 590), (1075, 1180), (0, 1170), (0, 700)],
    "IMG_0365": [(0, 1150), (1075, 1150), (1075, 1915), (0, 1915)],  # 腕が天板の上半分を覆うので下半分だけ
}


def frames(path, scale):
    """MOV を回転を適用した灰色のコマで順に返す（scale 倍に縮める）。"""
    w, h = probe(path)
    w, h = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-vf", f"scale={w}:{h},format=gray", "-f", "rawvideo", "-"],
                         stdout=subprocess.PIPE)
    while True:
        b = p.stdout.read(w * h)
        if len(b) < w * h:
            break
        yield np.frombuffer(b, np.uint8).reshape(h, w)
    p.wait()


def probe(path):
    """回転を適用した後の幅と高さ。"""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
                          "stream=width,height:stream_side_data=rotation", "-of", "csv=p=0", path],
                         capture_output=True, text=True, check=True).stdout
    v = [s for s in out.replace("\n", ",").split(",") if s]  # 例: "1920,1080,,-90"
    w, h = int(v[0]), int(v[1])
    return (h, w) if len(v) > 2 and abs(int(v[2])) in (90, 270) else (w, h)


def to3(m):
    return np.vstack([m, [0, 0, 1]])


def track(path, table, scale=0.5):
    """各コマについて、最初のコマの座標 → そのコマの座標 への 3×3 の変換を返す。"""
    poly = (np.array(table, np.float32) * scale).reshape(-1, 1, 2)
    Ts, ref, ref_pts, prev, T = [], None, None, None, np.eye(3)
    for g in frames(path, scale):
        if ref is None:
            ref = g
            m = np.zeros_like(g); cv2.fillPoly(m, [poly.astype(np.int32)], 255)
            ref_pts = cv2.goodFeaturesToTrack(g, 600, 0.01, 8, mask=m)
        else:
            # 1) 前のコマから追う
            m = np.zeros_like(g); cv2.fillPoly(m, [cv2.transform(poly, T[:2]).astype(np.int32)], 255)
            p0 = cv2.goodFeaturesToTrack(prev, 600, 0.01, 8, mask=m)
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, p0, None, winSize=(31, 31), maxLevel=4)
            ok = st[:, 0] == 1
            A, inl = cv2.estimateAffinePartial2D(p0[ok], p1[ok], method=cv2.RANSAC, ransacReprojThreshold=2)
            if A is not None:
                T = to3(A) @ T
            # 2) 最初のコマに合わせ直す（今の推定で最初のコマの向きに戻してから、残りのずれを測る）
            back = cv2.warpAffine(g, np.linalg.inv(T)[:2], (g.shape[1], g.shape[0]))
            q1, st, _ = cv2.calcOpticalFlowPyrLK(ref, back, ref_pts, None, winSize=(31, 31), maxLevel=3)
            ok = st[:, 0] == 1
            R, inl = cv2.estimateAffinePartial2D(ref_pts[ok], q1[ok], method=cv2.RANSAC, ransacReprojThreshold=2)
            if R is not None and inl.sum() > 60:
                T = T @ to3(R)
        Ts.append(T.copy()); prev = g
    S = np.diag([1 / scale, 1 / scale, 1]); Si = np.diag([scale, scale, 1])
    return np.array([S @ t @ Si for t in Ts])


def smooth(Ts, r=2):
    """推定の揺れを ±r コマの平均で抑える（カメラの本当の動きは数コマでは変わらない）。"""
    out = Ts.copy()
    for i in range(len(Ts)):
        out[i] = Ts[max(0, i - r):i + r + 1].mean(axis=0)
    return out


def render(path, out, Ts, keep, block, feather=31):
    """元の MOV を ffmpeg で読み、残す範囲の外をモザイクにして書く。マスクは Python から1コマずつ流す。"""
    w, h = probe(path)
    poly = np.array(keep, np.float32).reshape(-1, 1, 2)
    f = (f"[0:v]format=yuv444p,split[a][b];[b]scale=iw/{block}:ih/{block}:flags=area,"
         f"scale={w}:{h}:flags=neighbor[pix];[1:v]format=gray[m];[a][m]alphamerge[k];[pix][k]overlay=format=yuv444,format=yuv420p")
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-i", path,
                          "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
                          "-filter_complex", f, "-an", "-r", str(FPS), "-c:v", "libx264", "-preset", "medium", "-crf", "14",
                          "-map_metadata", "-1", out], stdin=subprocess.PIPE)
    for T in Ts:
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [cv2.transform(poly, T[:2]).astype(np.int32)], 255)
        p.stdin.write(cv2.GaussianBlur(m, (feather, feather), 0).tobytes())
    p.stdin.close(); p.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.expanduser("~/Downloads"))
    ap.add_argument("--out", default="outputs/video/masked")
    ap.add_argument("--only", nargs="*", default=TAKES)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for take in a.only:
        path = os.path.join(a.src, take + ".MOV")
        cache = os.path.join(a.out, take + "_track.npy")
        if os.path.exists(cache):
            Ts = np.load(cache)
        else:
            Ts = track(path, TABLE[take]); np.save(cache, Ts)
        print(take, len(Ts), "frames", flush=True)
        if take in KEEP:
            w, h = probe(path)
            render(path, os.path.join(a.out, take + ".mp4"), smooth(Ts), KEEP[take], block=round(32 * h / 1080))


if __name__ == "__main__":
    main()

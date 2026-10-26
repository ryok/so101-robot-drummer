"""カメラのオープン時間とfpsをアーム抜きで測る診断。
使い方: python scripts/camera_probe.py <index> [width height]
例: python scripts/camera_probe.py 1
    python scripts/camera_probe.py 2 424 240
AVFOUNDATIONバックエンドで開き、open所要時間・実測fps・MJPG設定可否を出す。
"""
import sys, time
import cv2

def probe(index, w, h, backend=cv2.CAP_AVFOUNDATION, backend_name="AVFOUNDATION"):
    print(f"--- index={index} {w}x{h} {backend_name} ---")
    t0 = time.time()
    cap = cv2.VideoCapture(index, backend)
    t_open = time.time() - t0
    print(f"VideoCapture() 所要: {t_open:.1f} 秒  isOpened={cap.isOpened()}")
    if not cap.isOpened():
        print("  開けませんでした"); return
    # MJPG設定を試す
    ok = cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_s = "".join(chr((fourcc >> 8*i) & 0xFF) for i in range(4))
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"MJPG set成功={ok}  実際のfourcc='{fourcc_s}'  実解像度={aw}x{ah}")
    # 実測fps（30フレーム）＋最後の有効フレームを保存
    n, t1, fails = 30, time.time(), 0
    last_frame = None
    for _ in range(n):
        ret, frame = cap.read()
        if not ret:
            fails += 1
        else:
            last_frame = frame
    dt = time.time() - t1
    print(f"{n}フレーム読み: {dt:.2f}秒  実測={n/dt:.1f} fps  失敗={fails}")
    if last_frame is not None and backend_name.startswith("AVF"):
        out = f"probe_index_{index}.jpg"
        cv2.imwrite(out, last_frame)
        print(f"  → 実画像を {out} に保存（このファイル名の番号がrecordのindex）")
    cap.release()

if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 640
    h = int(sys.argv[3]) if len(sys.argv) > 3 else 480
    # 両バックエンドで試す（find-cameras=既定ANY と record=AVFOUNDATION の差を切り分け）
    probe(idx, w, h, cv2.CAP_AVFOUNDATION, "AVFOUNDATION")
    print()
    probe(idx, w, h, cv2.CAP_ANY, "ANY(default)")

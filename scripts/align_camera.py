"""録画/rollout前に、俯瞰カメラの位置と照明が基準画像と揃っているかを確かめる。
使い方: python scripts/align_camera.py <index> [--ref refs/front_ref_mix.jpg] [--check]
例: python scripts/align_camera.py 0            # ライブで基準画像を半透明に重ねて表示
    python scripts/align_camera.py 0 --check    # 1枚撮って判定だけ出す（OKなら終了コード0）

判定: 位相相関で求めた平行移動量(px)と平均輝度の比。
  ずれ <= 5px かつ 輝度比 0.85〜1.15 → OK
アームやキューブは動くので、机・背景が画面の大半を占める前提の目安。
ライブ中のキー: q=終了 / m=表示切替(重ね合わせ↔エッジ↔差分) / +,-=透明度 / s=現在フレームを新しい基準として保存
"""
import argparse, os, sys, time
import cv2
import numpy as np

SHIFT_TOL_PX = 5.0
BRIGHT_RANGE = (0.85, 1.15)
MODES = ["blend", "edges", "diff"]


def measure(ref_gray, cur_gray):
    """基準→現在の平行移動量(dx, dy)と輝度比を返す。"""
    win = cv2.createHanningWindow(ref_gray.shape[::-1], cv2.CV_64F)
    (dx, dy), _ = cv2.phaseCorrelate(np.float64(ref_gray), np.float64(cur_gray), win)
    bright = cur_gray.mean() / max(ref_gray.mean(), 1e-6)
    return dx, dy, bright


def verdict(dx, dy, bright):
    shift = float(np.hypot(dx, dy))
    ok = shift <= SHIFT_TOL_PX and BRIGHT_RANGE[0] <= bright <= BRIGHT_RANGE[1]
    return ok, shift


def render(mode, ref, cur, alpha):
    if mode == "blend":
        return cv2.addWeighted(cur, 1 - alpha, ref, alpha, 0)
    if mode == "edges":
        # 現在=白、基準のエッジ=赤で重ねる。輪郭がぴったり重なれば位置が合っている
        out = cur.copy()
        edges = cv2.Canny(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), 60, 150)
        out[edges > 0] = (0, 0, 255)
        return out
    return cv2.applyColorMap(cv2.absdiff(ref, cur).max(axis=2), cv2.COLORMAP_INFERNO)


def open_camera(index, w, h):
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        sys.exit(f"index={index} を開けませんでした（camera_probe.py でindexを確認）")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    # 起動直後は露出が安定しないので数フレーム捨てる
    for _ in range(15):
        cap.read()
    return cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index", type=int)
    ap.add_argument("--ref", default="refs/front_ref_mix.jpg")
    ap.add_argument("--check", action="store_true", help="1枚だけ撮って判定して終了")
    args = ap.parse_args()

    ref = cv2.imread(args.ref)
    if ref is None and args.check:
        sys.exit(f"基準画像が読めません: {args.ref}（先にライブモードで s キーで作成してください）")
    if ref is not None:
        h, w = ref.shape[:2]
        ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    else:
        h, w = 480, 640  # 基準未作成: デフォルトサイズでライブ開始し s で作成
        ref_gray = None
        print(f"基準画像なし → ライブ表示中に s キーで {args.ref} を新規作成します")
    cap = open_camera(args.index, w, h)

    if args.check:
        ok_read, frame = cap.read()
        cap.release()
        if not ok_read or frame is None:
            sys.exit("フレームを取得できませんでした（フレーム0のカメラ）")
        frame = cv2.resize(frame, (w, h))
        dx, dy, bright = measure(ref_gray, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        ok, shift = verdict(dx, dy, bright)
        print(f"ずれ: dx={dx:+.1f} dy={dy:+.1f} ({shift:.1f}px)  輝度比: {bright:.2f}  → {'OK' if ok else 'NG'}")
        sys.exit(0 if ok else 1)

    mode, alpha = 0, 0.5
    while True:
        ok_read, frame = cap.read()
        if not ok_read or frame is None:
            time.sleep(0.05)
            continue
        frame = cv2.resize(frame, (w, h))
        if ref_gray is not None:
            dx, dy, bright = measure(ref_gray, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            ok, shift = verdict(dx, dy, bright)
            view = render(MODES[mode], ref, frame, alpha)
            color = (0, 200, 0) if ok else (0, 0, 255)
            cv2.putText(view, f"{'OK' if ok else 'NG'} shift={shift:.1f}px (dx={dx:+.0f},dy={dy:+.0f}) bright={bright:.2f}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            view = frame.copy()
            cv2.putText(view, "NO REF - press s to save current view as reference",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(view, f"mode={MODES[mode]}  q:quit m:mode +/-:alpha s:save-ref",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.imshow("align_camera", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("m"):
            mode = (mode + 1) % len(MODES)
        elif key in (ord("+"), ord("=")):
            alpha = min(alpha + 0.1, 1.0)
        elif key == ord("-"):
            alpha = max(alpha - 0.1, 0.0)
        elif key == ord("s"):
            os.makedirs(os.path.dirname(args.ref) or ".", exist_ok=True)
            cv2.imwrite(args.ref, frame)
            ref, ref_gray = frame.copy(), cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            print(f"基準画像を保存: {args.ref}")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

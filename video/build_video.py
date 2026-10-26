"""提出動画の下書きを ffmpeg で組み立てる（構成表は LIFE の ideas/2026-10/20261025-pai-final-video-plan.md）。
スマホの3テイクを小節の頭で切り替え、音は IMG_0344 を通しで使う（3本とも曲・譜面・テンポが同じなので1回の演奏に見える）。
字幕は PIL で PNG にして overlay する（この ffmpeg には drawtext が無い）。位置情報は -map_metadata -1 で消す。
使い方: python video/build_video.py [--src ~/Downloads] [--out outputs/video/draft.mp4]
"""
import argparse, os, subprocess

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

W, H, FPS = 1920, 1080, 30
BAR = 4 * 60 / 90                     # 90BPM の1小節（秒）
# 動画の音とログの打点列を合わせ、ボウルの小節のコマを目視して決めた「1小節目の頭」（動画の秒）
BAR1 = {"IMG_0344": -0.14, "IMG_0363": -3.54, "IMG_0365": -1.54}
PORTRAIT = {"IMG_0363", "IMG_0365"}
FONT_B = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
FONT_R = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
# 演奏ブロックの切り替え: (始めの小節, 小節数, テイク)
CUTS = [(1, 4, "IMG_0344"), (5, 4, "IMG_0363"), (9, 4, "IMG_0344"),
        (13, 4, "IMG_0365"), (17, 4, "IMG_0363"), (21, 5, "IMG_0344")]
TAIL = 2.6                            # 最後の小節のあとに余韻を残す秒数（21–24＋余韻で5小節分弱）


def bar_t(take, n):
    return BAR1[take] + (n - 1) * BAR


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"{' '.join(cmd[:6])} ...\n{r.stderr[-2000:]}")


def font(size, bold=True):
    return ImageFont.truetype(FONT_B if bold else FONT_R, size)


def text_png(path, lines, *, box=True, anchor="center", pos=None, bg=None):
    """lines: [(文字, 大きさ, 太字か, 色)]。bg があればその画像を暗くして背景にする。"""
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    if bg:
        b = ImageOps.fit(Image.open(bg).convert("RGB"), (W, H)).filter(ImageFilter.GaussianBlur(6))
        im = Image.blend(b, Image.new("RGB", (W, H)), 0.62).convert("RGBA")
    d = ImageDraw.Draw(im)
    sizes = [d.textbbox((0, 0), s, font=font(z, b)) for s, z, b, _ in lines]
    hs = [bb[3] - bb[1] + z * 0.55 for bb, (_, z, _, _) in zip(sizes, lines)]
    total, wmax = sum(hs), max(bb[2] - bb[0] for bb in sizes)
    if anchor == "center":
        x0, y0 = (W - wmax) / 2, (H - total) / 2
    else:                                  # 左下
        x0, y0 = pos[0], pos[1] - total
    if box and not bg:
        d.rounded_rectangle((x0 - 28, y0 - 22, x0 + wmax + 28, y0 + total + 4), 18, fill=(0, 0, 0, 165))
    y = y0
    for (s, z, b, c), h in zip(lines, hs):
        d.text((x0, y), s, font=font(z, b), fill=c)
        y += h
    im.save(path)


def video_in(take, start, dur, src):
    """テイクの一部を 1920×1080 の映像入力にする ffmpeg の引数とフィルタ。縦長は左右をぼかした同じ映像で埋める。"""
    args = ["-ss", f"{max(start, 0):.3f}", "-t", f"{dur:.3f}", "-i", os.path.join(src, take + ".MOV")]
    if take in PORTRAIT:
        f = (f"[0:v]split[a][b];[a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
             f"boxblur=30:2,eq=brightness=-0.12[bg];[b]scale=-2:{H}[fg];[bg][fg]overlay=(W-w)/2:0,fps={FPS},setsar=1[v0]")
    else:
        f = f"[0:v]scale={W}:{H},fps={FPS},setsar=1[v0]"
    return args, f


def segment(out, take, start, frames, src, overlays=()):
    """映像だけの区間を frames コマちょうどで作る（音は最後に1本で付ける。区間ごとに音を付けると継ぎ目でずれが積もる）。
    overlays: [(png, t0, t1)]"""
    dur = frames / FPS
    args, f = video_in(take, start, dur + 0.5, src)
    last = "v0"
    for k, (png, t0, t1) in enumerate(overlays):
        args += ["-loop", "1", "-t", f"{dur + 0.5:.3f}", "-i", png]
        f += f";[{last}][{k + 1}:v]overlay=0:0:enable='between(t,{t0:.3f},{t1:.3f})'[v{k + 1}]"
        last = f"v{k + 1}"
    run(["ffmpeg", "-v", "error", "-y", *args, "-filter_complex", f, "-map", f"[{last}]", "-frames:v", str(frames), "-an",
         "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-map_metadata", "-1", out])


def build_audio(out, pieces, src, song=None, gain_db=0.0):
    """pieces: [(テイク, 開始秒, 長さ秒, 音量)] を標本単位で長さをそろえてつなぎ、リミッターとフェードをかける。
    song: (wav, 動画上の位置秒, wav の頭を飛ばす秒, 長さ秒, 音量)。つないだ音に重ねてからリミッターをかける"""
    args, f, total = [], "", sum(d for _, _, d, _ in pieces)
    for k, (take, st, d, vol) in enumerate(pieces):
        args += ["-ss", f"{st:.3f}", "-t", f"{d + 0.2:.3f}", "-i", os.path.join(src, take + ".MOV")]
        f += (f"[{k}:a:0]aformat=sample_rates=48000:channel_layouts=stereo,volume={vol},apad,"
              f"atrim=end_sample={round(d * 48000)},asetpts=N/SR/TB,afade=t=in:d=0.3,afade=t=out:st={d - 0.3:.3f}:d=0.3[a{k}];")
    f += "".join(f"[a{k}]" for k in range(len(pieces))) + f"concat=n={len(pieces)}:v=0:a=1"
    if song:
        wav, at, skip, d, vol = song
        args += ["-i", wav]
        f += (f"[cam];[{len(pieces)}:a]aresample=48000,aformat=sample_rates=48000:channel_layouts=stereo,"
              f"atrim=start={skip:.4f}:duration={d:.4f},asetpts=N/SR/TB,volume={vol},afade=t=out:st={d - 2.5:.3f}:d=2.5,"
              f"adelay={round(at * 1000)}:all=1,apad[song];[cam][song]amix=inputs=2:duration=first:normalize=0")
    f += f",volume={gain_db}dB,"
    f += f"alimiter=limit=0.7:level=false,afade=t=in:d=1,afade=t=out:st={total - 2:.2f}:d=2[a]"
    run(["ffmpeg", "-v", "error", "-y", *args, "-filter_complex", f, "-map", "[a]", "-c:a", "pcm_s16le", out])


def still(src, take, t, path):
    run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t}", "-i", os.path.join(src, take + ".MOV"), "-frames:v", "1", path])


def still_seg(out, pngs, frames):
    """静止画（上に重ねる PNG があれば順に overlay）を frames コマの映像にする。"""
    cmd = ["ffmpeg", "-v", "error", "-y"]
    for png in pngs:
        cmd += ["-loop", "1", "-i", png]
    f, last = "", "0:v"
    for k in range(1, len(pngs)):
        f += f"[{last}][{k}:v]overlay[x{k}];"; last = f"x{k}"
    f += f"[{last}]fps={FPS},setsar=1,format=yuv420p[v]"
    run(cmd + ["-filter_complex", f, "-map", "[v]", "-frames:v", str(frames), "-c:v", "libx264", "-crf", "20",
               "-map_metadata", "-1", out])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.expanduser("~/Downloads"))
    ap.add_argument("--out", default="outputs/video/draft.mp4")
    ap.add_argument("--score", default="song/score.txt")
    ap.add_argument("--song", default=os.path.expanduser("~/Downloads/robot_song.wav"), help="Ableton から書き出した曲（無ければ重ねない）")
    ap.add_argument("--song-vol", type=float, default=0.5, help="重ねる曲の音量（倍率）")
    ap.add_argument("--gain", type=float, default=10.0, help="[dB] 全体の音量。打音のピークはリミッターで抑える")
    ap.add_argument("--cam-vol", type=float, default=1.0, help="曲を重ねるときの演奏ブロックのカメラ音の音量")
    a = ap.parse_args()
    work = os.path.join(os.path.dirname(a.out), "work")
    os.makedirs(work, exist_ok=True)
    p = lambda name: os.path.join(work, name)
    WHITE, SUB, ACC = (255, 255, 255, 255), (200, 210, 220, 255), (255, 214, 102, 255)

    # ---- 字幕・カードの画像 ----
    still(a.src, "IMG_0365", 40, p("bg_0365.jpg"))
    still(a.src, "IMG_0344", 30, p("bg_0344.jpg"))
    text_png(p("title.png"), [("AIロボットが 曲に合わせて 打楽器を叩く", 76, True, WHITE),
                              ("SO-101 × ACT × Ableton Link", 44, False, SUB)], bg=p("bg_0365.jpg"))
    # しくみ: video/diagram/system.png（Excalidraw）を白地の 1920×1080 に収める
    fig = Image.new("RGB", (W, H), (255, 255, 255))
    dia = ImageOps.contain(Image.open("video/diagram/system.png").convert("RGB"), (W - 120, H - 80), Image.LANCZOS)
    fig.paste(dia, ((W - dia.width) // 2, (H - dia.height) // 2))
    fig.save(p("how.png"))
    text_png(p("dev.png"), [("サンプルからの発展点", 52, True, ACC),
                            ("サンプル: 1つの物をつかんで置く", 34, False, SUB),
                            ("・楽譜の指示（drum / bowl）で、2つの楽器を叩き分ける", 38, False, WHITE),
                            ("・AIの照準と、固定の振り下ろしを組み合わせる", 38, False, WHITE),
                            ("・曲と拍単位で同期する（Ableton Link）", 38, False, WHITE),
                            ("・ACT（楽器ごとのモデル）と SmolVLA（指示文で1モデル）を比べる", 38, False, WHITE)], bg=p("bg_0344.jpg"))
    bars = [tok.split(":")[0] for tok in open(a.score).read().split()]
    for n, inst in enumerate(bars, 1):
        text_png(p(f"bar{n:02d}.png"), [(f"小節 {n}/24", 30, False, SUB),
                                        (f"\"Strike the {inst}\"", 44, True, ACC if inst == "bowl" else WHITE)],
                 anchor="left", pos=(70, 1020))
    text_png(p("song.png"), [("曲: 自作曲（90BPM）", 34, False, WHITE)], anchor="left", pos=(70, 110))
    text_png(p("result.png"), [("結果", 52, True, ACC),
                               ("3回の演奏とも、44打すべて楽器に命中（映像で確認）", 38, False, WHITE),
                               ("打点のずれ: 平均 +1〜3ms（曲なしでの基準測定）", 38, False, WHITE),
                               ("マイクの当たり判定は、伴奏があると空振りと誤判定した", 34, False, SUB),
                               ("→ 当たりは映像で検証した", 34, False, SUB)], bg=p("bg_0365.jpg"))
    # 手首カメラ: 学習時と、演奏の日（背景を戻した後）を並べる。照準 −8° だった朝（6:51）の wrist 画像は保存していない
    ref = Image.open("refs/wrist_ref_mix.jpg").convert("RGB").resize((720, 540))
    bad = Image.open("refs/wrist_perf_day.jpg").convert("RGB").resize((720, 540))
    cmp_ = Image.new("RGB", (W, H), (18, 20, 26))
    cmp_.paste(ref, (200, 300)); cmp_.paste(bad, (1000, 300))
    cmp_.save(p("cmp_bg.png"))
    text_png(p("cmp_txt.png"), [("比較と発見", 48, True, ACC),
                                ("推論時間 ACT 47ms / SmolVLA 466ms。狙う向きは同じ（+19°）だが SmolVLA はボウルの端に当たる", 30, False, WHITE),
                                ("ACT の照準は、机の外の背景（扇風機・箱）で変わった: −8° → 背景を戻して +13〜18°", 30, False, WHITE)],
             anchor="left", pos=(110, 250))
    cap = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(cap)
    d.text((200, 860), "学習データを録ったとき", font=font(32, False), fill=SUB)
    d.text((1000, 860), "演奏の日、背景を戻した後（照準 +13°）", font=font(32, False), fill=SUB)
    cap.save(p("cmp_cap.png"))
    text_png(p("limit.png"), [("限界", 48, True, ACC),
                              ("楽器の位置・照明・背景が変わると狙いがずれる", 38, False, WHITE),
                              ("（今回は条件を録画時に合わせて演奏した）", 32, False, SUB),
                              ("", 30, False, WHITE), ("ご視聴ありがとうございました", 44, True, WHITE)], bg=p("bg_0344.jpg"))

    # ---- 区間（映像はコマ数で、音は別に1本で） ----
    segs, audio = [], []
    def add(name, take, start, frames, overlays=(), sound=None):
        out = p(f"{len(segs):02d}_{name}.mp4"); segment(out, take, start, frames, a.src, overlays); segs.append(out)
        if sound:
            audio.append((sound[0], sound[1], frames / FPS, sound[2]))
    sec = lambda x: round(x * FPS)

    add("title", "IMG_0365", 40, sec(7), [(p("title.png"), 0, 7)], ("IMG_0365", 33, 0.35))
    still_seg(p(f"{len(segs):02d}_how.mp4"), [p("how.png")], sec(20)); segs.append(p(f"{len(segs):02d}_how.mp4"))
    audio.append(("IMG_0363", 2, 20.0, 0.4))
    add("dev", "IMG_0344", 30, sec(10), [(p("dev.png"), 0, 10)], ("IMG_0363", 22, 0.4))
    # 演奏: 音は 0344 を頭から1本で使い、映像は 0344 の時間軸で小節の頭（コマに丸めた位置）で切り替える。
    # 字幕は小節の頭の LEAD 秒前に替える（指示は打つ前に出ているので、少し早めのほうが映像と合って見える）
    LEAD = 0.25
    end = bar_t("IMG_0344", 25) + TAIL
    edges = [0.0] + [bar_t("IMG_0344", first) for first, _, _ in CUTS[1:]] + [end]
    fr = [sec(e) for e in edges]
    for (first, n, take), f0, f1 in zip(CUTS, fr, fr[1:]):
        t0 = f0 / FPS
        v0 = bar_t(take, 1) + t0 - bar_t("IMG_0344", 1)      # 同じ小節位置になるテイクの秒
        ov = []
        for b in range(first, min(first + n, 25)):
            s_, e_ = bar_t("IMG_0344", b) - t0 - LEAD, bar_t("IMG_0344", b + 1) - t0 - LEAD
            ov.append((p(f"bar{b:02d}.png"), max(s_, 0), e_ if b < 24 else 999))
        if first == 1:
            ov.append((p("song.png"), 0, 5))
        add(f"perf{first:02d}", take, v0, f1 - f0, ov)
    # 曲は Ableton から書き出した音（1小節目の頭が wav の0秒）を Link の拍＝ロボットの打点の格子に合わせて重ねる。
    # 撮影時のスピーカーは Link の拍より約0.18秒遅れて鳴っていた（0344 の音との相互相関と打点で測定）ので、
    # カメラ音に入っている曲は小さく残り、書き出した曲を大きく重ねる
    perf_at = sum(d for _, _, d, _ in audio)
    song = (a.song, perf_at, -bar_t("IMG_0344", 1), fr[-1] / FPS, a.song_vol) if os.path.exists(a.song) else None
    audio.append(("IMG_0344", 0.0, fr[-1] / FPS, a.cam_vol if song else 1.0))
    add("result", "IMG_0365", 20, sec(15), [(p("result.png"), 0, 15)], ("IMG_0365", 20, 0.3))
    # 比較: 静止画を映像にする
    out = p(f"{len(segs):02d}_cmp.mp4")
    still_seg(out, [p("cmp_bg.png"), p("cmp_txt.png"), p("cmp_cap.png")], sec(10))
    segs.append(out); audio.append(("IMG_0365", 35, 10.0, 0.3))
    add("limit", "IMG_0344", 60, sec(7), [(p("limit.png"), 0, 7)], ("IMG_0365", 45, 0.3))

    # ---- つなぐ ----
    lst = p("list.txt")
    with open(lst, "w") as f:
        f.writelines(f"file '{os.path.abspath(s)}'\n" for s in segs)
    build_audio(p("audio.wav"), audio, a.src, song, a.gain)
    run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-i", p("audio.wav"), "-map", "0:v", "-map", "1:a",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-map_metadata", "-1", "-movflags", "+faststart", "-shortest", a.out])
    frames = sum(int(subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v", "-show_entries",
                                     "stream=nb_read_packets", "-of", "csv=p=0", s], capture_output=True, text=True).stdout.strip().strip(",")) for s in segs)
    print(f"{a.out}  映像 {frames} コマ = {frames / FPS:.2f}秒 / 音 {sum(d for _, _, d, _ in audio):.2f}秒  区間 {len(segs)}")
    print("演奏ブロックの開始:", f"{sum(sec(x) for x in (7, 20, 10)) / FPS:.2f}秒")


if __name__ == "__main__":
    main()

"""ロボット用のループ曲を Ableton Live に組む（ableton-cli 経由。Live を起動し AbletonMCP の Remote Script を有効にしておく）。
キックとスネアは入れない。2拍目・4拍目のバックビートと小節頭の金属音はロボット（ドラム・ボウル）が担当する。
90BPM・Dマイナー・24小節（約64秒）を、各トラックのセッション枠0に1本の長いクリップとして置く。
構成: 1-4 導入（パッド＋ハット） / 5-12 A（＋鍵盤・ベース） / 13-20 B（＋ベル、ハット増） / 21-24 終わり（パッド＋鍵盤）
ロボットの譜面は score.txt（strike_conductor.py の --score にそのまま渡す）。
使い方: python song/build_song.py            （空のセットで実行。既存トラックは消さず末尾に足す）
"""
import json, os, subprocess, tempfile

BPM, BARS = 90, 24
BEATS = BARS * 4

# 1小節ずつのコード（ルートを除いた鍵盤のボイシング）とベースのルート
CHORDS = [  # Dm9, Bbmaj7, Gm9, A7sus4→A7
    dict(keys=[53, 57, 60, 64], root=38),
    dict(keys=[53, 57, 58, 62], root=34),
    dict(keys=[53, 57, 58, 62], root=31),
    dict(keys=[55, 57, 62, 64], keys2=[55, 57, 61, 64], root=33),
]
# ベルの4小節フレーズ（拍の裏に置き、ロボットの2・4拍とぶつけない）: (小節内の拍, 長さ, 音高)
BELL = [[(0.5, 1, 81), (2.5, 0.5, 77), (3, 1, 76)],
        [(0.5, 1.5, 74), (2.5, 1, 77)],
        [(0.5, 0.5, 79), (1.5, 0.5, 77), (2.5, 1.5, 74)],
        [(0.5, 1, 76), (2.5, 1, 73)]]


def n(p, t, d, v):
    return dict(pitch=p, start_time=round(t, 3), duration=d, velocity=v)


def section(bar):  # 0始まり
    return "intro" if bar < 4 else "A" if bar < 12 else "B" if bar < 20 else "outro"


def build():
    pad, keys, bass, bell, hats = [], [], [], [], []
    for b in range(BARS):
        s, c, t0 = section(b), CHORDS[b % 4], b * 4
        last = b == BARS - 1
        if last:
            c = CHORDS[0]
        pad += [n(p + 12, t0, 4, 55 if s != "B" else 65) for p in c["keys"]]
        if s in ("A", "B", "outro"):
            v = 72 if s != "outro" else 60
            keys += [n(p, t0, 1.5, v) for p in c["keys"]]
            if not last:
                keys += [n(p, t0 + 1.5, 0.5, v - 12) for p in c.get("keys2", c["keys"])]
            if b % 4 == 3 and not last:
                keys += [n(p, t0 + 3, 0.75, v - 8) for p in c.get("keys2", c["keys"])]
        if s in ("A", "B"):
            r = c["root"]
            bass += [n(r, t0, 1.5, 100), n(r + 12, t0 + 2.5, 0.5, 80), n(r, t0 + 3, 0.75, 90)]
        if s == "B":
            bell += [n(p, t0 + st, d, 70) for st, d, p in BELL[b % 4]]
        if s != "outro" or b < 22:
            vel = 45 if s == "intro" else 62 if s == "A" else 70
            hats += [n(42, t0 + k + 0.5, 0.25, vel) for k in range(4)]
            if s == "B":
                hats += [n(42, t0 + k + 0.75, 0.25, vel - 25) for k in range(4)]
                if b % 2 == 1:
                    hats.append(n(46, t0 + 3.5, 0.5, vel))
    return dict(Pad=pad, Keys=keys, Bass=bass, Bells=bell, Hats=hats)


def score():
    """ロボットの譜面（1小節16ステップ）。楽器を替える直前と直後の打点は2拍以上空ける
    （切替0.65秒＋前倒し約0.3秒が入る。10/25 実機で100BPM・2拍空きなら残り+90〜+140ms）。"""
    back = "....x.......x..."           # 2拍目と4拍目
    head = "x..............."           # 小節頭（ボウルの余韻を聞かせる）
    bars = [("drum", back)] * 8 + [("bowl" if i % 2 == 0 else "drum", back) for i in range(12)]
    bars[-1] = ("bowl", back)           # B の最後はボウルにして、終わりのボウル（小節頭）へ切替なしでつなぐ
    bars += [("bowl", head)] * 4
    assert len(bars) == BARS
    return " ".join(f"{i}:{p}" for i, p in bars)


SOUNDS = {
    "Pad": "query:Sounds#Pad:FileId_189432",                        # Analog Soft Pad
    "Keys": "query:Sounds#Piano%20&%20Keys:FileId_188193",          # E-Piano Classic
    "Bass": "query:Sounds#Bass:FileId_188501",                      # Basic Analog Bass
    "Bells": "query:Sounds#Mallets:FileId_188344",                  # Basic FM Bells
    "Hats": "query:Drums#FileId_188735",                            # 808 Core Kit
}
VOLUME_DB = {"Pad": -10, "Keys": -8, "Bass": -6, "Bells": -10, "Hats": -12}


def ab(*args):
    out = subprocess.run(["ableton", *map(str, args)], capture_output=True, text=True)
    if out.returncode:
        raise SystemExit(f"ableton {' '.join(map(str, args))}: {out.stderr or out.stdout}")
    return out.stdout


def main():
    parts = build()
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "score.txt"), "w") as f:
        f.write(score() + "\n")
    ab("tempo", BPM)
    first = json.loads(ab("session"))["track_count"]
    for k, (name, notes) in enumerate(parts.items()):
        t = first + k
        ab("track", "create")
        ab("track", "rename", t, name)
        ab("load", t, SOUNDS[name])
        ab("clip", "create", t, 0, "--length", BEATS)
        ab("clip", "rename", t, 0, f"robot song {name}")
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(notes, f)
        ab("clip", "add-notes", t, 0, "--file", f.name)
        ab("track", "volume", t, "--db", VOLUME_DB[name])
        print(f"track {t} {name}: {len(notes)} notes")
    print(f"{BPM}BPM {BARS}小節。譜面 → song/score.txt")


if __name__ == "__main__":
    main()

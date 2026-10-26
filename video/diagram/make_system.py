# 座標は手で決めたものをそのまま並べる（boilerplate だけ関数化）
import json, sys
E, seed = [], [100]
def base(t, id, x, y, w, h, stroke, fill="transparent", **kw):
    seed[0] += 1
    d = dict(type=t, id=id, x=x, y=y, width=w, height=h, strokeColor=stroke, backgroundColor=fill, fillStyle="solid",
             strokeWidth=kw.pop("sw", 2), strokeStyle=kw.pop("ss", "solid"), roughness=0, opacity=100, angle=0,
             seed=seed[0], version=1, versionNonce=seed[0], isDeleted=False, groupIds=[], boundElements=None,
             link=None, locked=False)
    d.update(kw); E.append(d); return d
def text(id, x, y, s, size, color, align="left", w=None, cont=None, ff=2):
    lines = s.split("\n")
    w = w or max(len(l) for l in lines) * size
    h = len(lines) * size * 1.25
    return base("text", id, x, y, w, h, color, text=s, originalText=s, fontSize=size, fontFamily=ff,
                textAlign=align, verticalAlign="middle" if cont else "top", containerId=cont, lineHeight=1.25, sw=1)
def box(id, x, y, w, h, stroke, fill, s, size, color="#1e293b", shape="rectangle", **kw):
    base(shape, id, x, y, w, h, stroke, fill, roundness={"type": 3}, boundElements=[{"id": id + "_t", "type": "text"}], **kw)
    lines = s.split("\n"); th = len(lines) * size * 1.25
    text(id + "_t", x + 10, y + (h - th) / 2, s, size, color, align="center", w=w - 20, cont=id)
def arrow(id, pts, color, sw=3, ss="solid"):
    x0, y0 = pts[0]
    rel = [[px - x0, py - y0] for px, py in pts]
    base("arrow", id, x0, y0, max(abs(p[0]) for p in rel), max(abs(p[1]) for p in rel), color, sw=sw, ss=ss,
         points=rel, startArrowhead=None, endArrowhead="arrow", startBinding=None, endBinding=None)
def line(id, pts, color, sw=3):
    x0, y0 = pts[0]; rel = [[px - x0, py - y0] for px, py in pts]
    base("line", id, x0, y0, max(abs(p[0]) for p in rel), max(abs(p[1]) for p in rel), color, sw=sw, points=rel)
def dot(id, cx, cy, r, color):
    base("ellipse", id, cx - r, cy - r, 2 * r, 2 * r, color, color, sw=1)

TITLE, SUB, BODY, DARK = "#312e81", "#4f46e5", "#64748b", "#1e293b"
# 見出し
text("title", 40, 20, "しくみ", 52, TITLE)
text("lead", 40, 92, "AIは演奏の前に「どこを狙うか」を決め、演奏中は拍に合わせて決まった動きで振り下ろす", 28, BODY)

# ① 演奏の前
text("l1", 40, 165, "① 演奏の前（楽器ごとに1回）", 34, SUB)
base("rectangle", "score", 40, 225, 360, 200, "#0f172a", "#0f172a", roundness={"type": 3})
text("score_h", 62, 240, "楽譜 score.txt", 24, "#94a3b8", ff=3)
text("score_b", 62, 285, "5:drum ....x.......x...\n9:bowl ....x.......x...", 22, "#34d399", w=330, ff=3)
text("leg", 62, 350, "x = 打つ（2拍目・4拍目）", 20, "#94a3b8", w=330)
text("cam", 62, 384, "＋カメラ2台・関節角6つ", 22, "#e2e8f0", w=330)
arrow("a1", [(400, 325), (465, 325)], "#3730a3")
box("ai", 470, 225, 420, 200, "#6d28d9", "#ddd6fe", "ACT（模倣学習）\n楽器ごとに1モデル\n推論 47ms", 32)
text("cmp", 470, 440, "比較: SmolVLA は1モデルに指示文\n\"Strike the bowl\" を入れる（466ms）", 24, BODY, w=440)
arrow("a2", [(890, 325), (945, 325)], "#6d28d9")
box("pose", 950, 225, 360, 200, "#047857", "#a7f3d0", "狙う姿勢\n楽器の真上で構え\n振り下ろした先", 32)
arrow("a3", [(1310, 325), (1365, 325)], "#047857")
box("cal", 1370, 225, 390, 200, "#b45309", "#fef3c7", "試し打ち 各3回\n当たる向き・深さ\n命令→当たる 約0.25秒", 30)

# ② 演奏中
text("l2", 40, 560, "② 演奏中（拍ごと）", 34, SUB)
box("link", 40, 680, 300, 160, "#c2410c", "#fed7aa", "Ableton Link\n90BPM\n曲と拍を共有", 28, shape="ellipse")
arrow("a4", [(340, 760), (420, 760)], "#c2410c")
line("tl", [(420, 760), (1760, 760)], "#64748b", sw=3)
for k, x in enumerate([480, 780, 1080, 1380, 1680]):
    hit = k in (1, 3)
    dot(f"b{k}", x, 760, 16 if hit else 9, "#047857" if hit else "#64748b")
    text(f"bn{k}", x - 14, 782, str(k % 4 + 1), 30, DARK, w=28, align="center")
    if hit:
        text(f"hit{k}", x - 45, 822, "当たる", 26, "#047857", w=90, align="center")
for k, x in enumerate([668, 1268]):
    dot(f"c{k}", x, 760, 11, "#c2410c")
    text(f"cn{k}", x - 26, 782, "命令", 26, "#c2410c", w=52, align="center")
line("br", [(668, 718), (668, 728), (780, 728), (780, 718)], "#c2410c", sw=2)
text("brt", 560, 676, "拍の約0.25秒前に命令", 26, "#c2410c", w=270, align="center")
# ①→② の合流
arrow("a5", [(1130, 425), (1262, 744)], "#047857", sw=2, ss="dashed")
arrow("a6", [(1500, 425), (1278, 744)], "#b45309", sw=2, ss="dashed")
text("a5t", 1070, 548, "姿勢", 26, "#047857", w=60)
text("a6t", 1418, 548, "前倒し量", 26, "#b45309", w=110)
# 下の注記
text("n1", 420, 880, "振り下ろしにはAIを使わない: 構え → 振り下ろした先 へ1回の命令で動かす", 26, BODY)
text("n2", 420, 922, "楽器が替わるとき: 肘を上げる → 横に回す → 下ろす（約0.6秒）", 26, BODY)

json.dump(dict(type="excalidraw", version=2, source="https://excalidraw.com", elements=E,
               appState=dict(viewBackgroundColor="#ffffff", gridSize=20), files={}),
          open(sys.argv[1], "w"), ensure_ascii=False, indent=1)

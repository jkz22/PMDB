"""Dancing cat under a rising/setting sun + animated progress bars of repo stats.

Numbers are a snapshot of 2026-10-04 from `git log` and `gh pr list` (see DEVIN_CARDS and OTHER_CARDS below).
Run: python scripts/make_repo_cat_video.py   (needs numpy, pillow, imageio, imageio-ffmpeg)
"""
import math
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS, SECS = 1280, 720, 24, 12
SS = 2  # cat supersampling
FONT = "/System/Library/Fonts/Helvetica.ttc"
f = lambda s, i=0: ImageFont.truetype(FONT, s, index=i)
INK, MUTED, BAR_BG = (245, 245, 240), (178, 178, 190), (60, 60, 76)
BLUE, ORANGE, GREEN, PINK, VIOLET = (57, 135, 229), (235, 104, 52), (27, 175, 122), (232, 123, 164), (150, 110, 230)
HORIZON = 560

# (label, value, suffix, colour): cards pop in one by one and count up from 0
DEVIN_CARDS = [
    ("PRs authored", 23, "", VIOLET), ("PRs reviewed", 97, "", VIOLET), ("lines added", 82887, "", VIOLET),
    ("commits co-authored", 44, "", VIOLET), ("review comments", 170, "", VIOLET), ("share of added lines", 33, "%", VIOLET),
]
OTHER_CARDS = [
    ("commits", 320, "", BLUE), ("pull requests", 103, "", BLUE), ("remote branches", 34, "", BLUE), ("tracked files", 1171, "", BLUE),
    ("days of work", 2, "", BLUE), ("PRs merged", 94, "", GREEN), ("PRs closed", 5, "", GREEN), ("PRs open", 4, "", GREEN),
    ("commits with Claude", 67, "", ORANGE), ("Claude share of commits", 33, "%", ORANGE),
]
LEGEND = [("repo", BLUE), ("pull requests", GREEN), ("Claude", ORANGE)]
LOGO = Image.open("outputs/clips/assets/devin_avatar.png").convert("RGB")  # official Devin mark (GitHub app avatar)


def ease(x):
    x = min(max(x, 0), 1)
    return 1 - (1 - x) ** 3


def back(x):
    """ease-out-back: pops slightly past 1 then settles."""
    x = min(max(x, 0), 1)
    c1, c3 = 1.70158, 2.70158
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2


def lerp(a, b, k):
    return tuple(int(a[i] + (b[i] - a[i]) * k) for i in range(3))


def sky(s):
    """s in [0,1] over the video: dawn -> noon -> dusk. Returns (img, sun_x, sun_y, elev)."""
    elev = math.sin(math.pi * s)  # 0 at horizon, 1 at noon
    top = lerp((16, 18, 44), (70, 140, 225), min(1, elev * 1.6))
    hor = lerp((255, 120, 60), (170, 215, 250), min(1, elev * 1.8))
    k = np.linspace(0, 1, HORIZON)[:, None, None]
    grad = (np.array(top) * (1 - k) + np.array(hor) * k).astype(np.uint8)
    arr = np.concatenate([np.repeat(grad, W, 1), np.full((H - HORIZON, W, 3), (24, 24, 34), np.uint8)], 0)
    img = Image.fromarray(arr)
    d = ImageDraw.Draw(img, "RGBA")
    if elev < .3:  # stars near dawn/dusk
        rng = np.random.default_rng(3)
        for _ in range(70):
            x, y = rng.integers(0, W), rng.integers(0, HORIZON - 160)
            d.ellipse([x, y, x + 2, y + 2], fill=(255, 255, 255, int(255 * (.3 - elev) / .3)))
    sx = W * (0.07 + 0.86 * s)
    sy = HORIZON + 40 - (HORIZON - 70) * elev
    return img, sx, sy, elev


def sun(img, sx, sy):
    """Glow + disc, clipped at the horizon so it sets behind the ground."""
    layer = Image.new("RGBA", (W, HORIZON), (0, 0, 0, 0))
    gd = ImageDraw.Draw(layer)
    for r, a in ((190, 20), (140, 32), (100, 50), (70, 80)):
        gd.ellipse([sx - r, sy - r, sx + r, sy + r], fill=(255, 205, 110, a))
    layer = layer.filter(ImageFilter.GaussianBlur(10))
    gd = ImageDraw.Draw(layer)
    for r, col in ((46, (255, 214, 120)), (38, (255, 232, 150)), (30, (255, 246, 200))):
        gd.ellipse([sx - r, sy - r, sx + r, sy + r], fill=col)
    out = img.convert("RGBA")
    out.alpha_composite(layer, (0, 0))
    return out.convert("RGB")


def shaded_ellipse(d, box, dark, light, light_off, steps=14):
    """Radial gradient approximation: nested ellipses from dark rim to light core, core shifted toward the light."""
    x0, y0, x1, y1 = box
    cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
    for i in range(steps):
        k = i / (steps - 1)
        sc = 1 - k * .78
        ox, oy = light_off[0] * k, light_off[1] * k
        c = lerp(dark, light, k ** 0.8)
        d.ellipse([cx + ox - rx * sc, cy + oy - ry * sc, cx + ox + rx * sc, cy + oy + ry * sc], fill=c)


def cat(t, light_dx):
    """A shaded tabby standing on hind legs and dancing. Returns an RGBA image (460 px)."""
    S = 460 * SS
    u = SS
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    beat = t * 2 * math.pi * 2
    bounce = abs(math.sin(beat / 2)) * 22 * u
    step = math.sin(beat / 2)
    cx, base = S / 2, S - 36 * u - bounce
    L = (light_dx * 30 * u, -26 * u)  # light direction follows the sun
    fur_d, fur_m, fur_l = (126, 74, 34), (196, 120, 56), (238, 172, 98)
    stripe, belly_d, belly_l = (84, 46, 22), (214, 182, 146), (250, 232, 206)
    rng = np.random.default_rng(5)

    # ground shadow
    sh = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(sh).ellipse([cx - 110 * u, S - 52 * u, cx + 110 * u, S - 22 * u], fill=(0, 0, 0, int(110 - bounce / u * 2)))
    sh = sh.filter(ImageFilter.GaussianBlur(8 * u))
    im.alpha_composite(sh)
    d = ImageDraw.Draw(im)

    # tail: thick, striped, swaying
    pts = [(cx + 62 * u + k * 8 * u, base - 50 * u - k * 9 * u + math.sin(beat + k * .55) * 18 * u) for k in range(16)]
    for k in range(len(pts) - 1):
        w = int((24 - k * .6) * u)
        d.line([pts[k], pts[k + 1]], fill=fur_m if k % 3 else stripe, width=w, joint="curve")
        d.ellipse([pts[k][0] - w / 2, pts[k][1] - w / 2, pts[k][0] + w / 2, pts[k][1] + w / 2], fill=fur_m if k % 3 else stripe)
    # hind legs
    for sgn, ph in ((-1, 0), (1, math.pi)):
        lift = max(0, math.sin(beat / 2 + ph)) * 24 * u
        x = cx + sgn * 40 * u
        shaded_ellipse(d, [x - 24 * u, base - 70 * u - lift, x + 24 * u, base + 18 * u - lift], fur_d, fur_m, (-sgn * 3 * u, -6 * u))
        shaded_ellipse(d, [x - 28 * u, base + 2 * u - lift, x + 28 * u, base + 30 * u - lift], fur_d, fur_l, (0, -4 * u))  # paw
        for tx in (-12, 0, 12):
            d.line([(x + tx * u, base + 14 * u - lift), (x + tx * u, base + 26 * u - lift)], fill=fur_d, width=2 * u)
    # body
    shaded_ellipse(d, [cx - 86 * u, base - 190 * u, cx + 86 * u, base + 4 * u], fur_d, fur_l, L)
    shaded_ellipse(d, [cx - 48 * u, base - 150 * u, cx + 48 * u, base - 8 * u], belly_d, belly_l, (L[0] * .4, L[1] * .4))
    for k in range(7):  # tabby stripes on both flanks: short slanted strokes
        y = base - 165 * u + k * 21 * u
        for sgn in (-1, 1):
            d.line([(cx + sgn * 83 * u, y), (cx + sgn * 58 * u, y + 11 * u)], fill=stripe, width=int(7 * u))
    for _ in range(160):  # fur strokes
        a, r = rng.uniform(0, 2 * math.pi), rng.uniform(.55, .98)
        px, py = cx + math.cos(a) * 86 * u * r, base - 93 * u + math.sin(a) * 96 * u * r
        ln = rng.uniform(5, 11) * u
        ang = a + rng.uniform(-.4, .4)
        d.line([(px, py), (px + math.cos(ang) * ln, py + math.sin(ang) * ln)], fill=fur_l if rng.random() < .5 else fur_d, width=max(1, u))
    # arms: alternate up/down, with paws
    for sgn, ph in ((-1, 0), (1, math.pi)):
        a = math.sin(beat / 2 + ph)
        sx, sy = cx + sgn * 76 * u, base - 140 * u
        ex, ey = sx + sgn * (42 + 22 * a) * u, sy - (26 + 62 * a) * u
        d.line([(sx, sy), (ex, ey)], fill=fur_m, width=int(26 * u))
        d.line([(sx, sy), (ex, ey)], fill=fur_l, width=int(10 * u))
        shaded_ellipse(d, [ex - 17 * u, ey - 17 * u, ex + 17 * u, ey + 17 * u], fur_d, fur_l, (-2 * u, -3 * u))
        for tx in (-6, 0, 6):
            d.line([(ex + tx * u, ey + 5 * u), (ex + tx * u, ey + 12 * u)], fill=fur_d, width=2 * u)
    # head
    hx, hy = cx + math.sin(beat / 2 + 1) * 12 * u, base - 232 * u + math.sin(beat) * 4 * u
    for sgn in (-1, 1):  # ears
        d.polygon([(hx + sgn * 76 * u, hy - 4 * u), (hx + sgn * 66 * u, hy - 92 * u), (hx + sgn * 14 * u, hy - 58 * u)], fill=fur_m)
        d.polygon([(hx + sgn * 62 * u, hy - 14 * u), (hx + sgn * 56 * u, hy - 70 * u), (hx + sgn * 26 * u, hy - 50 * u)], fill=(226, 140, 140))
        d.line([(hx + sgn * 66 * u, hy - 90 * u), (hx + sgn * 14 * u, hy - 58 * u)], fill=stripe, width=int(3 * u))
    shaded_ellipse(d, [hx - 82 * u, hy - 64 * u, hx + 82 * u, hy + 66 * u], fur_d, fur_l, (L[0] * .6, L[1] * .6))
    shaded_ellipse(d, [hx - 40 * u, hy + 8 * u, hx + 40 * u, hy + 62 * u], belly_d, belly_l, (0, -3 * u))  # muzzle
    for k in (-1, 0, 1):  # forehead "M" stripes
        d.line([(hx + k * 18 * u, hy - 62 * u), (hx + k * 14 * u, hy - 26 * u)], fill=stripe, width=int(6 * u))
    for sgn in (-1, 1):  # cheek stripes
        for k in range(2):
            d.line([(hx + sgn * 80 * u, hy + (2 + k * 14) * u), (hx + sgn * 52 * u, hy + (8 + k * 12) * u)], fill=stripe, width=int(5 * u))
    for sgn in (-1, 1):  # eyes: iris gradient, slit pupil, highlight
        ex, ey = hx + sgn * 34 * u, hy - 6 * u
        d.ellipse([ex - 20 * u, ey - 16 * u, ex + 20 * u, ey + 16 * u], fill=(28, 22, 18))
        shaded_ellipse(d, [ex - 17 * u, ey - 14 * u, ex + 17 * u, ey + 14 * u], (70, 110, 40), (196, 224, 90), (0, -3 * u), steps=8)
        d.ellipse([ex - 5 * u, ey - 13 * u, ex + 5 * u, ey + 13 * u], fill=(10, 10, 10))
        d.ellipse([ex + 4 * u, ey - 10 * u, ex + 10 * u, ey - 4 * u], fill=(255, 255, 255))
        d.arc([ex - 22 * u, ey - 20 * u, ex + 22 * u, ey + 14 * u], 200, 340, fill=(40, 24, 14), width=int(3 * u))
    d.polygon([(hx - 9 * u, hy + 24 * u), (hx + 9 * u, hy + 24 * u), (hx, hy + 36 * u)], fill=(214, 120, 128))  # nose
    d.line([(hx, hy + 36 * u), (hx, hy + 44 * u)], fill=(90, 50, 40), width=int(2.5 * u))
    d.arc([hx - 24 * u, hy + 36 * u, hx, hy + 58 * u], 20, 160, fill=(90, 50, 40), width=int(2.5 * u))
    d.arc([hx, hy + 36 * u, hx + 24 * u, hy + 58 * u], 20, 160, fill=(90, 50, 40), width=int(2.5 * u))
    for sgn in (-1, 1):  # curved whiskers
        for k in (-1, 0, 1):
            wx0, wy0 = hx + sgn * 36 * u, hy + (30 + k * 6) * u
            wp = [(wx0 + sgn * j * 10 * u, wy0 + k * j * 4 * u - math.sin(j * .35) * 6 * u) for j in range(9)]
            d.line(wp, fill=(250, 250, 250, 230), width=max(1, int(1.6 * u)))

    im = im.resize((S // u, S // u), Image.LANCZOS)
    return im.rotate(math.sin(beat / 2) * 5, center=(S // u // 2, S // u - 40), resample=Image.BICUBIC)


def panel(img, box, alpha=118):
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(ov).rounded_rectangle(box, 18, fill=(12, 12, 20, alpha))
    return Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")


def frame(i):
    t = i / FPS
    s = t / SECS
    img, sx, sy, elev = sky(s)
    # dance floor
    d = ImageDraw.Draw(img)
    for gx in range(6):
        for gy in range(2):
            ph = (gx + gy + int(t * 4)) % 4
            col = [BLUE, ORANGE, GREEN, PINK][ph]
            d.rectangle([20 + gx * 85, 585 + gy * 62, 20 + gx * 85 + 80, 585 + gy * 62 + 57], fill=tuple(c // 3 for c in col))
    img = sun(panel(img, (540, 140, W - 24, 680)), sx, sy)  # sky -> panel -> sun, then the cat in front
    c = cat(t, (sx - 280) / W)
    img.paste(c, (46, 150), c)
    d = ImageDraw.Draw(img)
    # notes
    for k in range(4):
        p = (t * .6 + k / 4) % 1
        x, y = 80 + k * 100 + math.sin(p * 6) * 20, 330 - p * 220
        col = [BLUE, ORANGE, GREEN, PINK][k]
        d.ellipse([x, y + 22, x + 22, y + 36], fill=col)
        d.line([(x + 21, y + 29), (x + 21, y)], fill=col, width=4)
        d.line([(x + 21, y), (x + 36, y + 10)], fill=col, width=4)
    # title + tiles
    d.text((46, 30), "PMDB repo progress", font=f(46, 1), fill=INK)
    d.text((46, 90), "snapshot 2026-10-04 · git + GitHub", font=f(22), fill=(235, 235, 240))
    px0, pw = 560, 676

    def card(x, y, w, h, lab, val, suf, col, t0, vfont, lfont=15):
        pop = back((t - t0) / .45)
        if pop <= 0:
            return
        cx_, cy_ = x + w / 2, y + h / 2
        hw, hh = w * min(pop, 1.08) / 2, h * min(pop, 1.08) / 2
        d.rounded_rectangle([cx_ - hw, cy_ - hh, cx_ + hw, cy_ + hh], 14, fill=(12, 12, 20))
        if pop < .95:
            return
        d.rounded_rectangle([x, y, x + w, y + 6], 3, fill=col)
        n = round(val * ease((t - t0 - .2) / 1.3))
        txt = f"{n:,}{suf}"
        d.text((cx_, y + h * .42), txt, font=f(vfont if len(txt) < 6 else int(vfont * .85), 1), fill=INK, anchor="mm")
        d.text((cx_, y + h * .80), lab, font=f(lfont), fill=MUTED, anchor="mm")

    # Devin banner: official mark + name + roles
    bp = back((t - .3) / .5)
    if bp > 0:
        d.rounded_rectangle([px0, 156, px0 + pw, 232], 16, fill=(24, 18, 44), outline=VIOLET, width=3)
        if bp > .6:
            sz = int(60 * min(bp, 1.1))
            badge = Image.new("RGB", (60, 60), (255, 255, 255))
            badge.paste(LOGO.resize((60, 60), Image.LANCZOS))
            mask = Image.new("L", (60, 60), 0)
            ImageDraw.Draw(mask).rounded_rectangle([0, 0, 59, 59], 14, fill=255)
            badge = badge.resize((sz, sz), Image.LANCZOS)
            mask = mask.resize((sz, sz), Image.LANCZOS)
            img.paste(badge, (px0 + 18 + (60 - sz) // 2, 164 + (60 - sz) // 2), mask)
            d.text((px0 + 100, 172), "Devin AI", font=f(34, 1), fill=INK)
            d.text((px0 + 100, 208), "PR author  ·  PR reviewer  ·  commit co-author", font=f(17), fill=(200, 190, 240))
    for k, (lab, val, suf, col) in enumerate(DEVIN_CARDS):
        r, c_ = divmod(k, 3)
        card(px0 + c_ * 230, 244 + r * 104, 216, 94, lab, val, suf, col, .9 + k * .4, 38)
    # everything else, smaller
    for k, (lab, val, suf, col) in enumerate(OTHER_CARDS):
        r, c_ = divmod(k, 5)
        card(px0 + c_ * 137, 460 + r * 82, 128, 72, lab, val, suf, col, 3.7 + k * .25, 28, 12)
    lx = px0
    for name, col in LEGEND:
        d.ellipse([lx, 636, lx + 14, 650], fill=col)
        d.text((lx + 22, 634), name, font=f(18), fill=INK)
        lx += 44 + 11 * len(name)
    return np.asarray(img)


if __name__ == "__main__":
    out = "outputs/clips/repo_progress_cat.mp4"
    with imageio.get_writer(out, fps=FPS, codec="libx264", quality=8, macro_block_size=None) as w:
        for i in range(FPS * SECS):
            w.append_data(frame(i))
    for name, sec in (("dawn", 1.0), ("mid", 3.5), ("noon", 8.0), ("dusk", 11.0)):
        Image.fromarray(frame(int(FPS * sec))).save(f"/tmp/cat_{name}.png")
    print(out)

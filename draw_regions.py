#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式圈定"国家/区域"脚本 —— 配合 asr_demo.py 的分块自适应演示
================================================================
在模拟景观上用鼠标圈出若干区域（多边形），保存为 regions.npy，
然后运行:
    python asr_demo.py --blocks regions.npy

操作说明（弹出的窗口内）:
    左键单击 : 添加多边形顶点
    右键单击 : 闭合当前多边形（成为一块区域）
    按 c     : 撤销上一个顶点；若当前无未闭合顶点，则删除上一块
    按 q     : 保存并退出

注意:
    - 圈出的区域会作为"国家"，小区域会被 asr_demo.py 自动合并到邻近块
      （质心最近邻，对应完整版行政分块 strategy B 的小国合并）。
    - 留白（未圈区域）为 -1，不会被用作分块。
    - 需要图形界面（Windows 上直接运行即可；无显示环境请用默认块）。
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from asr_demo import gen_landscape, GENERATIONS


def main():
    ap = argparse.ArgumentParser(description='交互式圈定区域')
    ap.add_argument('--grid', type=int, default=200)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    n = args.grid
    rng = np.random.default_rng(args.seed)
    _, _, _, _, p_true, _ = gen_landscape(n, rng, GENERATIONS['g_clean'])

    labels = np.full((n, n), -1, dtype=int)
    polys = []          # 已闭合的多边形（顶点列表）
    cur = []            # 正在画的顶点
    artists = []        # 图元句柄（重绘用）

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(p_true, cmap='viridis', origin='lower')
    ax.set_title('左键加点 | 右键闭合 | c 撤销 | q 保存退出', fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046)

    def redraw():
        for a in artists:
            a.remove()
        artists.clear()
        labels[:] = -1
        pts = np.stack(np.mgrid[0:n, 0:n], -1).reshape(-1, 2)
        for k, verts in enumerate(polys):
            mask = Path(verts).contains_points(pts).reshape(n, n)
            labels[mask] = k
            xs = [v[0] for v in verts] + [verts[0][0]]
            ys = [v[1] for v in verts] + [verts[0][1]]
            artists.append(ax.plot(xs, ys, '-r', lw=2)[0])
        if cur:
            xs = [v[0] for v in cur]
            ys = [v[1] for v in cur]
            artists.append(ax.plot(xs, ys, '--b', lw=1.5)[0])
            artists.append(ax.plot(xs, ys, 'bo', ms=4)[0])
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes is None or event.xdata is None:
            return
        if event.button == 1:
            cur.append((event.xdata, event.ydata))
            redraw()
        elif event.button == 3:
            if len(cur) >= 3:
                polys.append(cur.copy())
                cur.clear()
                redraw()

    def on_key(event):
        if event.key == 'c':
            if cur:
                cur.clear()
            elif polys:
                polys.pop()
            redraw()
        elif event.key == 'q':
            redraw()
            plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    n_regions = len(np.unique(labels)) - (1 if -1 in np.unique(labels) else 0)
    np.save('regions.npy', labels)
    print(f'已保存 regions.npy（{n}×{n}，圈出 {n_regions} 块区域）')
    print('运行: python asr_demo.py --blocks regions.npy')


if __name__ == '__main__':
    main()

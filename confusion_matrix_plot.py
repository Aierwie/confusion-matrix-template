"""
confusion_matrix_plot.py

A styled confusion matrix plotting utility.

Author: Xiaowu Guo
Date: 2026-03-24

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib import font_manager as fm
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap
# =========================
# 1. 你的真实标签和预测标签
#    如果你已经有 y_true / y_pred，就直接替换这里
# =========================


def plot_confusion_matrix(y_true, y_pred, model_name, class_names):
   
    cmat = confusion_matrix(y_true, y_pred)

    fontsize=9
    # 类别名称
   

    # =========================
    # 2. 自动选择字体
    # =========================
    candidate_fonts = [
        "Comic Sans MS",
        "Segoe Print",
        "Bradley Hand ITC",
        "Chalkboard",
        "Marker Felt",
        "DejaVu Sans"
    ]

    available_fonts = {f.name for f in fm.fontManager.ttflist}
    chosen_font = "DejaVu Sans"
    for f in candidate_fonts:
        if f in available_fonts:
            chosen_font = f
            break

    plt.rcParams["font.family"] = chosen_font
    plt.rcParams["axes.unicode_minus"] = False

    # =========================
    # 3. 颜色映射：对角线red，错误项blue
    # =========================
    n = cmat.shape[0]

    diag_vals = np.diag(cmat)
    off_vals = cmat.copy()
    np.fill_diagonal(off_vals, 0)

    diag_max = max(diag_vals.max(), 1)
    off_max = max(off_vals.max(), 1)

    def light_blues():
        return LinearSegmentedColormap.from_list(
            "light_blues",
            ["#f4f7fb", "#cfd8e3", "#9fb3c8"]  # 浅 → 中 → 稍深（但不刺眼）
        )

    # ====== 柔和红色（正确） ======
    def light_reds():
        return LinearSegmentedColormap.from_list(
            "light_reds",
            ["#fff5f5", "#f8caca", "#ef8f8f"]  # 柔和红
        )

    diag_cmap = light_reds()   # 正确 → 红
    off_cmap = light_blues()   # 错误 → 蓝

    diag_norm = Normalize(vmin=0, vmax=diag_max * 0.8)
    off_norm = Normalize(vmin=0, vmax=off_max * 0.8)

    # =========================
    # 4. 创建画布
    # =========================
    fig = plt.figure(figsize=(10, 7), dpi=150, facecolor="#f7f7f7")
    ax = fig.add_axes([0.08, 0.12, 0.68, 0.72], facecolor="#f7f7f7")

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.invert_yaxis()
    ax.set_aspect("equal")

    for spine in ax.spines.values():
        spine.set_visible(False)

    # =========================
    # 5. 画圆角格子 + 数字圆圈
    # =========================
    box_size = 0.78
    offset = (1 - box_size) / 2

    for i in range(n):
        for j in range(n):
            val = cmat[i, j]

            # 颜色和边框规则
            if i == j:
                facecolor = diag_cmap(diag_norm(val))
                edgecolor = "#dc2626" if val == diag_vals.max() else "none"
                lw = 2.2 if val == diag_vals.max() else 0
            else:
                facecolor = off_cmap(off_norm(val))
                edgecolor = "black" if val == 0 else "none"
                lw = 2.0 if val == 0 else 0

            shadow = FancyBboxPatch(
            (j + offset-0.03 , i + offset-0.03 ),  
            box_size+0.06, box_size+0.06,
            boxstyle="round,pad=0.02,rounding_size=0.14",
            linewidth=0,
            facecolor="black",
            alpha=0.06,   # ⭐ 阴影强度（关键）
            zorder=1      # ⭐ 在底层
            )
            ax.add_patch(shadow)
            # 圆角方块
            rect = FancyBboxPatch(
                (j + offset, i + offset),
                box_size,
                box_size,
                boxstyle="round,pad=0.02,rounding_size=0.12",
                linewidth=lw,
                edgecolor=edgecolor,
                facecolor=facecolor,
                zorder=2
            )
            ax.add_patch(rect)

            # 数字小圆圈
            circ = Circle(
                (j + 0.5, i + 0.5),
                0.018 * fontsize,
                facecolor="white",
                edgecolor="#333333",
                linewidth=1.1,
                zorder=3
            )
            ax.add_patch(circ)

            ax.text(
                j + 0.5,
                i + 0.52,
                str(val),
                ha="center",
                va="center",
                fontsize=9,
                color="#c81e1e",
                fontweight="bold"
            )

    # =========================
    # 6. 坐标轴标签
    # =========================
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.1)
    ax.set_xticklabels(class_names, rotation=20, ha="right", fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)
    ax.tick_params(length=0)

    ax.set_xlabel("Predicted Class", fontsize=14, fontweight="bold", labelpad=8)
    ax.set_ylabel("True Class", fontsize=14, fontweight="bold", labelpad=18)

    # =========================
    # 7. 主标题
    # =========================
    fig.text(
        0.42,
        0.90,
        f"Classification Confusion Matrix\n(Model: {model_name})",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold"
    )

    # =========================
    # 8. 右侧双图例条
    # =========================
    # 正确分类
    cax1 = fig.add_axes([0.68, 0.50, 0.03, 0.30], facecolor="#f7f7f7")
    gradient1 = np.linspace(0, 1, 256).reshape(-1, 1)
    cax1.imshow(gradient1, aspect="auto", cmap=diag_cmap, origin="lower")
    cax1.set_xticks([])
    cax1.set_yticks([0, 255])
    cax1.set_yticklabels(["0", str(diag_max)], fontsize=10)
    for s in cax1.spines.values():
        s.set_visible(False)

    fig.text(
        0.69,
        0.82,
        "Correct\nPredictions",
        color="#b91c1c",
        fontsize=11,
        ha="center",
        fontweight="bold"
    )

    # 错误分类
    cax2 = fig.add_axes([0.68, 0.10, 0.03, 0.30], facecolor="#f7f7f7")
    gradient2 = np.linspace(0, 1, 256).reshape(-1, 1)
    cax2.imshow(gradient2, aspect="auto", cmap=off_cmap, origin="lower")
    cax2.set_xticks([])
    cax2.set_yticks([0, 255])
    cax2.set_yticklabels(["0", str(off_max)], fontsize=10)
    for s in cax2.spines.values():
        s.set_visible(False)

    fig.text(
        0.69,
        0.42,
        "Confusion\nErrors",
        color="#1d4ed8",
        fontsize=11,
        ha="center",
        fontweight="bold"
    )

    # =========================
    # 9. 可选：虚线外框，更像示意图
    # =========================
    # border_ax = fig.add_axes([0.04, 0.04, 0.90, 0.88], facecolor="none")
    # for spine in border_ax.spines.values():
    #     spine.set_linestyle((0, (3, 3)))
    #     spine.set_linewidth(1.2)
    #     spine.set_edgecolor("#cfcfcf")
    # border_ax.set_xticks([])
    # border_ax.set_yticks([])
    # border_ax.patch.set_alpha(0)

    plt.show()
"""评测指标 —— 纯函数，不依赖网络，可单独测试。

约定：每条记录是一个 dict，至少包含
    p    : float  模型给"它自己选中的那个答案"的概率
    ok   : bool   是否判对
    conf : float  用来当开关的那个分数（choice/score 用 SDK 的 confidence；noul 用 max(p,1-p)）
"""

from __future__ import annotations

import math
import statistics
from typing import Iterable, Sequence


def accuracy(rows: Sequence[dict]) -> float:
    return sum(r["ok"] for r in rows) / len(rows) if rows else float("nan")


def brier(rows: Sequence[dict], key: str = "p") -> float:
    """概率预测的均方误差。0 = 完美；0.25 = 二分类瞎猜。"""
    if not rows:
        return float("nan")
    return statistics.fmean((r[key] - (1.0 if r["ok"] else 0.0)) ** 2 for r in rows)


def _bin_edges(nbins: int) -> list[tuple[float, float]]:
    return [(i / nbins, (i + 1) / nbins) for i in range(nbins)]


def _bucket(rows, lo, hi, key, last):
    return [r for r in rows if lo <= r[key] < hi or (last and r[key] >= hi)]


def reliability(rows: Sequence[dict], key: str = "p", nbins: int = 10) -> list[dict]:
    """可靠性曲线：每个概率区间里，平均概率 vs 实际准确率。"""
    edges = _bin_edges(nbins)
    out = []
    for i, (lo, hi) in enumerate(edges):
        b = _bucket(rows, lo, hi, key, i == nbins - 1)
        if not b:
            continue
        acc = accuracy(b)
        mp = statistics.fmean(r[key] for r in b)
        out.append({"lo": lo, "hi": hi, "n": len(b), "acc": acc, "mean_p": mp, "gap": mp - acc})
    return out


def ece(rows: Sequence[dict], key: str = "p", nbins: int = 10) -> float:
    """期望校准误差：各区间 |平均概率 - 实际准确率| 按样本数加权。越小越好。"""
    if not rows:
        return float("nan")
    return sum(b["n"] / len(rows) * abs(b["gap"]) for b in reliability(rows, key, nbins))


def mce(rows: Sequence[dict], key: str = "p", nbins: int = 10, min_n: int = 5) -> float:
    """最大校准误差：最差的那个区间偏多少。只看样本数够的区间。"""
    bs = [b for b in reliability(rows, key, nbins) if b["n"] >= min_n]
    return max((abs(b["gap"]) for b in bs), default=float("nan"))


def sweep(rows: Sequence[dict], key: str = "conf",
          thresholds: Iterable[float] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)) -> list[dict]:
    """阈值扫描：卡在某个分数以上自动处理，覆盖多少量、错多少。"""
    out = []
    for th in thresholds:
        auto = [r for r in rows if r[key] >= th]
        if not auto:
            continue
        out.append({
            "threshold": th,
            "coverage": len(auto) / len(rows),
            "accuracy": accuracy(auto),
            "errors": sum(not r["ok"] for r in auto),
            "deferred": len(rows) - len(auto),
        })
    return out


def recommend(rows: Sequence[dict], target_acc: float = 0.99,
              key: str = "conf", step: float = 0.01) -> dict | None:
    """找到能达到目标准确率的最低阈值 —— 即在保证质量的前提下最大化自动化率。"""
    best = None
    th = 0.0
    while th <= 1.0 + 1e-9:
        auto = [r for r in rows if r[key] >= th]
        if auto and accuracy(auto) >= target_acc:
            best = {"threshold": round(th, 4), "coverage": len(auto) / len(rows),
                    "accuracy": accuracy(auto), "errors": sum(not r["ok"] for r in auto),
                    "deferred": len(rows) - len(auto), "n_auto": len(auto)}
            break
        th += step
    return best


def confusion(rows: Sequence[dict]) -> dict[tuple, int]:
    """混淆矩阵：(gold, pred) -> 次数。只对离散答案有意义。"""
    c: dict[tuple, int] = {}
    for r in rows:
        k = (str(r.get("gold")), str(r.get("pred")))
        c[k] = c.get(k, 0) + 1
    return c


def stability(repeats: Sequence[Sequence[dict]]) -> dict:
    """多次重复同一批样本，衡量抖动。

    repeats[i] 是第 i 轮的结果，各轮顺序一致、一一对应。
    """
    if len(repeats) < 2:
        return {}
    n = len(repeats[0])
    p_std, flips = [], 0
    for j in range(n):
        ps = [rep[j]["p"] for rep in repeats]
        preds = {str(rep[j]["pred"]) for rep in repeats}
        p_std.append(statistics.pstdev(ps))
        flips += len(preds) > 1
    return {
        "runs": len(repeats), "n": n,
        "mean_p_std": statistics.fmean(p_std),
        "max_p_std": max(p_std),
        "argmax_flip_rate": flips / n,
        "flipped_items": flips,
    }


def near_threshold(rows: Sequence[dict], th: float, band: float = 0.05,
                   key: str = "conf") -> list[dict]:
    """落在阈值附近的样本 —— 这些是抖动会真正改变决策的那批。"""
    return [r for r in rows if abs(r[key] - th) <= band]


# ---------------------------------------------------------------- self-check
def _selfcheck() -> None:
    """用构造出的已知答案验证指标实现本身。"""
    # 完美校准：概率 p 的那批，恰好 p 比例是对的
    rows = []
    for p, n in ((0.6, 100), (0.9, 100)):
        k = round(p * n)
        rows += [{"p": p, "conf": p, "ok": True}] * k
        rows += [{"p": p, "conf": p, "ok": False}] * (n - k)
    assert ece(rows) < 1e-9, ece(rows)
    assert abs(brier(rows) - (0.6*0.4 + 0.9*0.1)/2) < 1e-9

    # 系统性虚高：全报 0.9，实际只有一半对
    bad = [{"p": 0.9, "conf": 0.9, "ok": i % 2 == 0} for i in range(100)]
    assert abs(ece(bad) - 0.4) < 1e-9, ece(bad)

    # 阈值推荐
    mix = ([{"p": 0.99, "conf": 0.99, "ok": True}] * 50 +
           [{"p": 0.6, "conf": 0.6, "ok": False}] * 50)
    rec = recommend(mix, target_acc=1.0)
    assert rec and rec["coverage"] == 0.5 and rec["errors"] == 0, rec

    # 稳定性
    r1 = [{"p": 0.8, "pred": "a"}, {"p": 0.5, "pred": "a"}]
    r2 = [{"p": 0.82, "pred": "a"}, {"p": 0.5, "pred": "b"}]
    st = stability([r1, r2])
    assert st["argmax_flip_rate"] == 0.5 and st["flipped_items"] == 1

    print("metrics.py 自检通过")


if __name__ == "__main__":
    _selfcheck()

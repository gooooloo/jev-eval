"""confidence 到底是不是摆设 —— 可证伪版。

两个独立问题：
  Q1 概率准不准？ 用 p_max（模型给选中答案的概率）算 ECE / Brier。
  Q2 confidence 能不能当开关？ 扫阈值，看自动化率 vs 自动化部分的准确率。
"""

import asyncio
import statistics
import time

from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul

from genset import build

SEM = asyncio.Semaphore(20)


async def run_item(client, it):
    q = (
        Choice(instructions=it["instructions"], criteria=it["criteria"])
        if it["kind"] == "choice"
        else Noul(instructions=it["instructions"])
    )
    async with SEM:
        for attempt in range(3):
            try:
                resp = await client.system_one(it["state"], {"q": q})
                break
            except Exception as e:
                if attempt == 2:
                    return None
                await asyncio.sleep(1.5 * (attempt + 1))

    if it["kind"] == "choice":
        a = resp.choices["q"]
        pred, p_max, conf = a.choice, max(a.probabilities.values()), a.confidence
    else:
        a = resp.nouls["q"]
        pred = a.noul >= 0.5
        p_max = max(a.noul, 1 - a.noul)
        conf = p_max  # Noul 没有独立 confidence，概率本身就是全部信息

    return {**it, "pred": pred, "p_max": p_max, "conf": conf, "ok": pred == it["gold"],
            "tokens": resp.usage.input_tokens}


def ece(rows, key="p_max", nbins=10):
    """期望校准误差：各桶 |平均概率 - 实际准确率| 按样本数加权。0 = 完美校准。"""
    tot, n = 0.0, len(rows)
    for i in range(nbins):
        lo, hi = i / nbins, (i + 1) / nbins
        b = [r for r in rows if lo <= r[key] < hi or (i == nbins - 1 and r[key] == 1.0)]
        if not b:
            continue
        tot += len(b) / n * abs(statistics.mean(r[key] for r in b) - sum(r["ok"] for r in b) / len(b))
    return tot


def brier(rows, key="p_max"):
    """Brier score：概率预测的均方误差。越低越好，0.25 = 瞎猜。"""
    return statistics.mean((r[key] - (1.0 if r["ok"] else 0.0)) ** 2 for r in rows)


def table(rows, key, title, edges):
    print(f"\n{title}")
    print(f"  {'区间':<14} {'n':>4} {'实际准确率':>10} {'平均'+key:>12} {'偏差':>8}")
    for lo, hi in edges:
        b = [r for r in rows if lo <= r[key] < hi or (hi >= 1.0 and r[key] == 1.0)]
        if not b:
            continue
        acc = sum(r["ok"] for r in b) / len(b)
        mp = statistics.mean(r[key] for r in b)
        flag = "  <-- 虚高" if mp - acc > 0.12 else ("  <-- 偏保守" if acc - mp > 0.12 else "")
        print(f"  [{lo:.2f},{hi:.2f})  {len(b):>4} {acc:>9.1%} {mp:>12.3f} {mp-acc:>+8.3f}{flag}")


async def main():
    items = build(n_per_tier=20, seed=7)
    print(f"样本 {len(items)} 条（choice/noul × tier 1-4 × 20）")

    async with AsyncTypeSafeClient() as client:
        t0 = time.perf_counter()
        rows = [r for r in await asyncio.gather(*(run_item(client, it) for it in items)) if r]
        wall = time.perf_counter() - t0

    print(f"完成 {len(rows)}/{len(items)}，耗时 {wall:.1f}s，"
          f"{len(rows)/wall:.1f} req/s，input tokens {sum(r['tokens'] for r in rows)}")

    print("\n" + "=" * 72)
    print("准确率（按难度和类型）")
    print(f"  {'tier':<6} {'choice':>18} {'noul':>18}")
    for t in (1, 2, 3, 4):
        cell = []
        for k in ("choice", "noul"):
            b = [r for r in rows if r["tier"] == t and r["kind"] == k]
            cell.append(f"{sum(r['ok'] for r in b)}/{len(b)} = {sum(r['ok'] for r in b)/len(b):>5.0%}" if b else "-")
        print(f"  tier {t}  {cell[0]:>18} {cell[1]:>18}")
    print(f"  总体   {sum(r['ok'] for r in rows)}/{len(rows)} = {sum(r['ok'] for r in rows)/len(rows):.1%}")

    print("\n" + "=" * 72)
    print("Q1  概率准不准？")
    for label, sub in (("全部", rows),
                       ("choice", [r for r in rows if r["kind"] == "choice"]),
                       ("noul", [r for r in rows if r["kind"] == "noul"])):
        print(f"  {label:<8} ECE={ece(sub):.4f}   Brier={brier(sub):.4f}   n={len(sub)}")
    print("  参考：ECE<0.05 算校准良好；LLM 自报置信度通常 0.15~0.30")

    edges = [(0.0,.5),(.5,.6),(.6,.7),(.7,.8),(.8,.9),(.9,.99),(.99,1.01)]
    table(rows, "p_max", "可靠性曲线（p_max vs 实际准确率）", edges)

    print("\n" + "=" * 72)
    print("Q2  confidence 能不能当开关？（仅 choice，noul 的 conf 就是概率）")
    ch = [r for r in rows if r["kind"] == "choice"]
    print(f"  {'阈值':<8} {'自动化率':>10} {'自动部分准确率':>14} {'漏网错误数':>10} {'转人工数':>9}")
    for th in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
        auto = [r for r in ch if r["conf"] >= th]
        if not auto:
            continue
        bad = sum(not r["ok"] for r in auto)
        print(f"  >={th:<6.2f} {len(auto)/len(ch):>9.1%} {sum(r['ok'] for r in auto)/len(auto):>13.1%} "
              f"{bad:>10} {len(ch)-len(auto):>9}")

    print("\n" + "=" * 72)
    print("最危险的样本：高 confidence 但判错")
    bad = sorted([r for r in rows if not r["ok"]], key=lambda r: -r["conf"])[:6]
    for r in bad:
        print(f"  conf={r['conf']:.3f} p_max={r['p_max']:.3f} tier={r['tier']} {r['kind']}"
              f"  pred={r['pred']} gold={r['gold']}")
    if not bad:
        print("  无错误")


if __name__ == "__main__":
    asyncio.run(main())

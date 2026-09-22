"""同一批题，两种问法对打。

Choice: 一个四选一问题。
Noul  : 拆成 N 个「X 是不是实付最高的？」，取概率最大者。
两者在同一个 state 上，唯一差别是 primitive 的选择。
"""
import asyncio, statistics
from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul
from genset import build

SEM = asyncio.Semaphore(15)

async def both(c, it):
    people = list(it["criteria"])
    async with SEM:
        r_ch, r_nl = await asyncio.gather(
            c.system_one(it["state"], {"q": Choice(instructions=it["instructions"],
                                                   criteria=it["criteria"])}),
            c.system_one(it["state"], {
                p: Noul(instructions=f"在本期账单记录里，{p} 是实付金额最高的人吗？")
                for p in people}),
        )
    a = r_ch.choices["q"]
    nv = {p: r_nl.nouls[p].noul for p in people}
    top = max(nv, key=nv.get)
    ranked = sorted(nv.values(), reverse=True)
    return {
        "gold": it["gold"], "tier": it["tier"],
        "ch_pred": a.choice, "ch_conf": a.confidence,
        "ch_p": max(a.probabilities.values()), "ch_ok": a.choice == it["gold"],
        "nl_pred": top, "nl_p": nv[top], "nl_ok": top == it["gold"],
        "nl_margin": ranked[0] - (ranked[1] if len(ranked) > 1 else 0),
        "nl_sum": sum(nv.values()),
        "ch_tok": r_ch.usage.input_tokens, "nl_tok": r_nl.usage.input_tokens,
    }

def ece(rows, pk, okk, nbins=10):
    tot = 0.0
    for i in range(nbins):
        lo, hi = i/nbins, (i+1)/nbins
        b = [r for r in rows if lo <= r[pk] < hi or (i == nbins-1 and r[pk] == 1.0)]
        if b:
            tot += len(b)/len(rows)*abs(statistics.mean(r[pk] for r in b) - sum(r[okk] for r in b)/len(b))
    return tot

async def main():
    items = [i for i in build(20, seed=7) if i["kind"] == "choice"]
    async with AsyncTypeSafeClient() as c:
        rows = await asyncio.gather(*(both(c, it) for it in items))

    print(f"同一批 {len(rows)} 道题，两种问法\n" + "=" * 62)
    print(f"  {'':<10} {'Choice':>12} {'Noul 拆解':>12}")
    print(f"  {'准确率':<10} {sum(r['ch_ok'] for r in rows)/len(rows):>11.1%} "
          f"{sum(r['nl_ok'] for r in rows)/len(rows):>12.1%}")
    print(f"  {'ECE':<10} {ece(rows,'ch_p','ch_ok'):>11.4f} {ece(rows,'nl_p','nl_ok'):>12.4f}")
    print(f"  {'input tok':<10} {sum(r['ch_tok'] for r in rows):>11} {sum(r['nl_tok'] for r in rows):>12}")

    print("\n  按难度：")
    for t in (1,2,3,4):
        b = [r for r in rows if r["tier"] == t]
        print(f"    tier {t}   choice {sum(r['ch_ok'] for r in b)}/{len(b)}"
              f"      noul {sum(r['nl_ok'] for r in b)}/{len(b)}")

    print("\n" + "=" * 62)
    print("两者分歧的题（谁对？）")
    dis = [r for r in rows if r["ch_pred"] != r["nl_pred"]]
    print(f"  分歧 {len(dis)}/{len(rows)} 题；其中 choice 对 {sum(r['ch_ok'] for r in dis)}，"
          f"noul 对 {sum(r['nl_ok'] for r in dis)}")

    print("\n" + "=" * 62)
    print("Noul 拆解丢掉的东西：概率和不受约束")
    s = [r["nl_sum"] for r in rows]
    print(f"  N 个 noul 概率之和：min {min(s):.2f}  median {statistics.median(s):.2f}  max {max(s):.2f}")
    print(f"  （Choice 恒等于 1.00；noul 之和可以 >1 或 <1，即可能'都是'或'都不是'）")
    bad = [r for r in rows if r["nl_sum"] < 0.6 or r["nl_sum"] > 1.6]
    print(f"  明显偏离 1 的有 {len(bad)}/{len(rows)} 题")

    print("\n" + "=" * 62)
    print("当成开关用：两者的阈值表")
    for name, pk, okk in (("Choice conf", "ch_conf", "ch_ok"), ("Noul margin", "nl_margin", "nl_ok")):
        print(f"  {name}")
        for th in (0.5, 0.7, 0.8, 0.9):
            auto = [r for r in rows if r[pk] >= th]
            if auto:
                print(f"    >={th}  自动化 {len(auto)/len(rows):>5.1%}  准确 "
                      f"{sum(r[okk] for r in auto)/len(auto):>6.1%}  漏错 {sum(not r[okk] for r in auto)}")

asyncio.run(main())

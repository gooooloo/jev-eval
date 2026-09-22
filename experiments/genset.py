"""程序化生成带确定答案的样本。

关键：ground truth 由生成逻辑决定，不依赖任何人的主观判断。
难度分 4 档，目的是让模型在高档位上真的出错 —— 否则画不出校准曲线。
"""

import random

NAMES = ["陈嘉树", "林晚", "赵柏川", "沈知微", "周砚", "吴落雁", "郑允"]
PLANS = {"基础版": 39, "标准版": 99, "专业版": 299, "旗舰版": 899}


def _people(rng, n):
    return rng.sample(NAMES, n)


# ---------- Choice：谁的实付金额最高 ----------

def choice_item(rng, tier):
    """实付 = 单价 × 席位 × (1 - 折扣)，按年付再 ×12×0.8。答案由算术唯一确定。"""
    n = {1: 3, 2: 3, 3: 4, 4: 4}[tier]
    people = _people(rng, n)
    rows, totals = [], {}

    for p in people:
        plan = rng.choice(list(PLANS))
        seats = rng.randint(1, 12)
        price = PLANS[plan]

        if tier == 1:
            total = price * seats
            rows.append(f"{p}：{plan}，{seats} 个席位，月付，无折扣。")
        elif tier == 2:
            disc = rng.choice([0, 0.1, 0.2])
            total = price * seats * (1 - disc)
            rows.append(f"{p}：{plan}，{seats} 个席位，月付，折扣 {int(disc*100)}%。")
        elif tier == 3:
            disc = rng.choice([0, 0.1, 0.15, 0.2])
            yearly = rng.choice([True, False])
            total = price * seats * (1 - disc) * (12 * 0.8 if yearly else 1)
            cycle = "年付（年付在折后价基础上再打八折，按 12 个月计）" if yearly else "月付"
            rows.append(f"{p}：{plan}，{seats} 个席位，{cycle}，折扣 {int(disc*100)}%。")
        else:
            disc = rng.choice([0, 0.05, 0.1, 0.15, 0.2, 0.25])
            yearly = rng.choice([True, False])
            trial = rng.random() < 0.35
            total = price * seats * (1 - disc) * (12 * 0.8 if yearly else 1)
            if trial:
                total = 0.0
            cycle = "年付（年付在折后价基础上再打八折，按 12 个月计）" if yearly else "月付"
            extra = "，目前处于免费试用期，本期实付为 0" if trial else ""
            rows.append(f"{p}：{plan}，{seats} 个席位，{cycle}，折扣 {int(disc*100)}%{extra}。")

        totals[p] = round(total, 2)

    ranked = sorted(totals.items(), key=lambda kv: -kv[1])
    # 并列则重生成，保证答案唯一
    if len(ranked) > 1 and abs(ranked[0][1] - ranked[1][1]) < 0.01:
        return None

    rng.shuffle(rows)
    pricelist = "、".join(f"{k} {v} 元/席位/月" for k, v in PLANS.items())
    state = f"套餐单价：{pricelist}。\n\n本期账单记录：\n" + "\n".join(rows)
    return {
        "kind": "choice",
        "tier": tier,
        "state": state,
        "instructions": "根据账单记录，谁本期的实付金额最高？",
        "criteria": {p: None for p in people},
        "gold": ranked[0][0],
        "debug": totals,
    }


# ---------- Noul：一条陈述是否与记录相符 ----------

def noul_item(rng, tier):
    people = _people(rng, {1: 2, 2: 3, 3: 4, 4: 5}[tier])
    facts, seats_map = [], {}
    for p in people:
        plan = rng.choice(list(PLANS))
        seats = rng.randint(1, 20)
        active = rng.random() < 0.7
        seats_map[p] = (plan, seats, active)
        st = "已激活" if active else "已停用"
        facts.append(f"{p}：{plan}，{seats} 个席位，账号{st}。")

    rng.shuffle(facts)
    state = "团队成员记录：\n" + "\n".join(facts)
    target = rng.choice(people)
    plan, seats, active = seats_map[target]
    truth = rng.random() < 0.5  # 一半真一半假

    if tier <= 2:
        # 直接查表
        if truth:
            claim = f"{target} 的套餐是{plan}。"
        else:
            claim = f"{target} 的套餐是{rng.choice([x for x in PLANS if x != plan])}。"
    elif tier == 3:
        # 需要比较
        total_active = sum(s for _, s, a in seats_map.values() if a)
        if truth:
            claim = f"所有已激活账号的席位总数是 {total_active}。"
        else:
            claim = f"所有已激活账号的席位总数是 {total_active + rng.choice([-3,-2,-1,1,2,3])}。"
    else:
        # 否定 + 范围 + 排除
        others = [p for p in people if p != target]
        cnt = sum(1 for p in others if seats_map[p][1] > seats)
        if truth:
            claim = f"除 {target} 外，恰好有 {cnt} 个人的席位数比 {target} 多。"
        else:
            wrong = cnt + rng.choice([-2, -1, 1, 2])
            if wrong < 0:
                wrong = cnt + 1
            claim = f"除 {target} 外，恰好有 {wrong} 个人的席位数比 {target} 多。"

    return {
        "kind": "noul",
        "tier": tier,
        "state": state + f"\n\n待核实的陈述：{claim}",
        "instructions": "上述「待核实的陈述」与团队成员记录相符吗？",
        "criteria": None,
        "gold": truth,
        "debug": claim,
    }


def build(n_per_tier=20, seed=7):
    rng = random.Random(seed)
    items = []
    for tier in (1, 2, 3, 4):
        made = 0
        while made < n_per_tier:
            it = choice_item(rng, tier)
            if it:
                items.append(it)
                made += 1
        for _ in range(n_per_tier):
            items.append(noul_item(rng, tier))
    return items


if __name__ == "__main__":
    items = build(2)
    for it in items[:2] + items[-2:]:
        print("=" * 60)
        print(f"[tier {it['tier']} {it['kind']}] gold={it['gold']}")
        print(it["state"])
        print("Q:", it["instructions"])

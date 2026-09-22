# Jev 评测框架

用你自己的标注数据，测出 Jev 在**你的任务**上的准确率、校准质量和可用阈值。

数据全程留在本机。除了调用 TypeSafe API（评测本身必需），不向任何地方发送。

## 快速开始

```bash
export TYPESAFE_API_KEY=...

python jeveval.py selftest                 # 先用合成数据验证框架能跑通
python jeveval.py run --spec spec.json --data data.jsonl --out raw.jsonl
python jeveval.py report --raw raw.jsonl   # 换阈值重新分析，不再调 API
```

## 两个输入文件

### `spec.json` — 问题定义（所有样本共用）

```json
{
  "questions": {
    "team": {
      "type": "choice",
      "instructions": "这条工单应该路由给哪个团队？",
      "criteria": {
        "billing":   "扣款、退款、发票",
        "technical": "报错、功能失效",
        "sales":     "购买咨询、升级套餐"
      }
    },
    "urgent": {
      "type": "noul",
      "instructions": "这条工单需要在 2 小时内响应吗？"
    },
    "severity": {
      "type": "score",
      "instructions": "影响范围有多大？",
      "criteria": ["个别用户", "部分团队", "全员不可用"]
    }
  }
}
```

`criteria` 写成 `{选项: 说明}` 比 `{选项: null}` 效果好得多 —— 说明会进模型，
选项 key 只是给代码用的。Score 的档位必须是能独立看懂的具体描述。

### `data.jsonl` — 标注数据（每行一条）

```jsonl
{"id": "t-001", "state": "我被扣了两次钱...", "gold": {"team": "billing", "urgent": true}}
{"id": "t-002", "state": {"title": "登录失败", "body": "...", "plan": "专业版"}, "gold": {"team": "technical"}}
```

- `state` 可以是字符串，也可以是 JSON 对象（内容有多个部分时用对象，官方推荐）
- `gold` 只需标你关心的问题；没标的问题会跳过，不影响其他指标
- `gold` 取值：
  - choice → 选项 key；也可以给**列表**表示「这几个都算对」
  - noul → `true` / `false`
  - score → 档位标签，或数值（数值按 ±0.5 容差判对）
- 想让某条用不同的问题，在该行加 `"questions": {...}` 覆盖 spec

## 报告怎么读

| 指标 | 含义 | 怎么算好 |
|---|---|---|
| **准确率** | 判对的比例 | 看你的业务要求 |
| **ECE** | 期望校准误差：模型说的概率和实际准确率差多少 | < 0.05 好；> 0.10 会告警，说明概率不可信 |
| **MCE** | 最差那个区间偏多少 | 揭示局部失准，均值掩盖不了 |
| **Brier** | 概率预测的均方误差 | 越低越好，0.25 ≈ 二分类瞎猜 |
| **可靠性曲线** | 各概率区间：模型说的 vs 实际 | **偏差为正 = 虚高（危险）**；为负 = 偏保守（安全） |
| **阈值扫描** | 卡某个分数以上自动处理，覆盖多少、错多少 | 找自动化率和错误数的平衡点 |
| **▶ 推荐阈值** | 达到目标准确率的**最低**阈值 | 直接抄进你的代码 |
| **混淆矩阵** | 哪两个类目最容易搞混 | 通常意味着 criteria 描述需要区分得更清楚 |
| **稳定性** | 多轮重复的概率标准差和结论翻转率 | 翻转率 > 5% 说明不少样本卡在决策边界 |

## 典型用法

```bash
# 换目标准确率重新推荐阈值（不调 API）
python jeveval.py report --raw raw.jsonl --target 0.95

# 测稳定性：同一批跑 5 轮，看结论翻转率
python jeveval.py run --spec spec.json --data data.jsonl --out raw.jsonl --repeat 5

# 固定模型版本 —— 上线前必做，阈值是和版本绑定的
python jeveval.py run --spec spec.json --data data.jsonl --out raw.jsonl --model jev-1.13.0

# 先跑 20 条试水，确认格式没问题再全量
python jeveval.py run --spec spec.json --data data.jsonl --out raw.jsonl --limit 20
```

## 三条容易踩的坑

1. **校准只在「题目有解」时成立。** 如果 `state` 里缺了判断所需的信息，模型照样会给出
   高 confidence 的答案 —— 它没法告诉你「你少给了字段」。输入完整性只能靠代码保证。

2. **阈值不可移植。** 换任务、改 criteria 措辞、升级模型版本，都要重测。所以上线时用
   `--model jev-1.13.0` 固定版本，别用 `jev-latest`。

3. **按问题形状选 primitive，不是按「谁更准」。** 问自己「这几个答案能同时为真吗？」——
   不能 → Choice（归一化带来的互斥约束是有用信息）；能 → 每个标签一个 Noul。
   把互斥问题拆成多个 Noul 会丢掉约束，实测更差。

## 文件

- `jeveval.py` — CLI：`run` / `report` / `selftest`
- `metrics.py` — 纯函数指标，`python metrics.py` 可单独自检
- `examples/` — 数据格式样例（`selftest` 会在 `_selftest/` 另行生成一份）

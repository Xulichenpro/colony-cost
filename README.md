
# 2026 MCM Problem B: Colony Cost Optimization

本项目旨在建立数学模型，优化从2050年开始向火星（或其他目的地）运输1亿吨物资的成本与时间。

## 1. 优化模型设计
### 目标
*   **目标函数1**：最小化总成本 (Minimize Total Cost)
*   **目标函数2**：最小化运输时间 (Minimize Transport Time)

### 约束与参数
1.  **总运量**：1亿吨（100 million tons）。
2.  **运输方式**：
    *   **火箭 (Rockets)**：
        *   发射频次：最大 10次/天。
        *   单次运载量：150吨 (150,000 kg)。
        *   成本计算：动态预测（详见下文）。
    *   **太空电梯 (Space Elevators)**：
        *   数量：3台。
        *   容量：每台每年最多 179,000 吨。
        *   成本：每年固定维护费 $100,000（忽略通过电梯的边际运输成本）。
3.  **预测模型**：火箭制造成本和燃料价格均随时间动态变化。

## 2. 火箭发射成本计算 (Rocket Launch Cost)
单次发射的总成本通过 `calculate_total_cost.py` 计算，公式如下：

$$ \text{Cost} = M_{load} \times \left[ \frac{C_{mfg}}{N} + R_{prop} \times (0.3 \times P_{fuel} + 0.7 \times P_{ox}) + M_{maint} \right] $$

**参数解释**：
*   **$C_{mfg}$ (Construction Cost)**: 火箭制造成本 ($/kg)。基于历史数据（含SpaceX数据）采用指数衰减模型预测，随技术成熟逐年降低。
*   **$N$ (Reuses)**: 火箭复用次数，默认为 20 次。这是降低成本的关键因素。
*   **$P_{fuel}$ (Fuel Price)**: 燃料价格 ($/kg)。基于历史数据通过统计模型预测。
*   **$R_{prop}$ (Propellant Ratio)**: 推进剂占火箭总重的比例，取 91%。
*   **$P_{ox}$**: 氧化剂等其他推进剂成本，视为常数 ($0.15/kg)。
*   **$M_{maint}$**: 维护成本，定为制造成本的 1%。

## 3. 模型输出
模型 (`optimization_model.py`) 遍历不同的火箭发射策略（0-10次/天），输出：
*   **Pareto前沿**：展示成本与时间的权衡关系。
*   **运输方案**：各年份火箭与太空电梯的运量分配。
*   **关键指标**：总耗时、总投入资金。

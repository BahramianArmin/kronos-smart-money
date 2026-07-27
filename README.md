# Kronos × Smart Money
# Hybrid AI + ICT trading strategy
#
# - model/  — Kronos foundation model for financial K-lines
# - ict/    — ICT concepts: FVG, Order Block, Liquidity, MSS, CISD
# - tests/  — Unit tests for all components

## Quick Start
```bash
pip install -r requirements.txt
# Download Kronos-small model
# Then run tests
python tests/test_ict.py
```

## Hybrid Strategy
1. **Kronos** — pre-trained transformer predicts future OHLCV tokens
2. **ICT** — detects Fair Value Gaps, Order Blocks, liquidity sweeps, MSS
3. **HybridScorer** — combines both: 60% ICT score + 40% Kronos confidence + alignment bonus

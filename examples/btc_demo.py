"""
Kronos × Smart Money — Example: daily BTCUSDT analysis
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd

# ICT concepts
from ict import score_ict, detect_fvg, detect_liquidity, detect_mss, detect_cisd


def run_demo():
    """Demo: generate synthetic BTC data and run ICT analysis."""
    print("=" * 60)
    print("Kronos × Smart Money — Demo Analysis")
    print("=" * 60)

    # 1. Generate synthetic BTC daily data
    np.random.seed(42)
    n = 200
    dates = pd.date_range('2024-01-01', periods=n, freq='D')
    close = 40000 + np.cumsum(np.random.randn(n) * 500)
    high = close + np.abs(np.random.randn(n)) * 800
    low = close - np.abs(np.random.randn(n)) * 800
    open_ = close - np.random.randn(n) * 300
    volume = np.random.randint(10000, 50000, n)

    df = pd.DataFrame({
        'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume
    }, index=dates)

    print(f"\n📊 Data: {len(df)} daily candles (BTCUSDT)")
    print(f"   Range: {df.index[0].date()} → {df.index[-1].date()}")
    print(f"   Price: ${df['close'].iloc[0]:.0f} → ${df['close'].iloc[-1]:.0f}")

    # 2. Detect ICT concepts
    print(f"\n{'='*40}")
    print("🔍 ICT Concept Detection")
    print(f"{'='*40}")

    fvgs = detect_fvg(df)
    print(f"\n📐 FVGs: {len(fvgs)} detected")
    if fvgs:
        bullish = sum(1 for f in fvgs if f.type == 'bullish')
        bearish = len(fvgs) - bullish
        print(f"   Bullish: {bullish} | Bearish: {bearish}")

    liq = detect_liquidity(df, lookback=15)
    print(f"\n💧 Liquidity zones: {len(liq)}")
    swing_highs = sum(1 for l in liq if l['type'] == 'swing_high')
    swing_lows = len(liq) - swing_highs
    print(f"   Swing Highs: {swing_highs} | Swing Lows: {swing_lows}")

    mss_list = detect_mss(df, lookback=15)
    print(f"\n🔄 MSS/BOS: {len(mss_list)}")
    bullish_mss = sum(1 for m in mss_list if m.type == 'bullish_mss')
    bearish_mss = len(mss_list) - bullish_mss
    print(f"   Bullish MSS: {bullish_mss} | Bearish MSS: {bearish_mss}")

    cisd_list = detect_cisd(df)
    print(f"\n🔗 CISD sequences: {len(cisd_list)}")
    for c in cisd_list:
        print(f"   {c.direction} × {c.count} (imbalance: {c.total_imbalance:.1f})")

    # 3. ICT Scoring
    print(f"\n{'='*40}")
    print("📊 ICT Scoring")
    print(f"{'='*40}")

    scores = score_ict(df, lookback_liquidity=15)
    buy_signals = [s for s in scores if s.signal == 'buy']
    sell_signals = [s for s in scores if s.signal == 'sell']
    strong_signals = [s for s in scores if s.composite_score >= 7]

    print(f"\nSignals:")
    print(f"   Buy:  {len(buy_signals)}")
    print(f"   Sell: {len(sell_signals)}")
    print(f"   Neutral: {len(scores) - len(buy_signals) - len(sell_signals)}")
    print(f"   High conviction (≥7): {len(strong_signals)}")

    if strong_signals:
        best = max(scores, key=lambda s: s.composite_score)
        print(f"\n🏆 Best signal:")
        print(f"   Index: {best.idx} | Date: {best.timestamp}")
        print(f"   Signal: {best.signal} | Score: {best.composite_score}")
        print(f"   FVGs: {best.fvg_count} | OBs: {best.ob_count}")
        print(f"   Liquidity Swept: {best.liquidity_swept} | MSS: {best.mss_active}")

    print(f"\n{'='*40}")
    print("✅ Analysis complete")
    print(f"{'='*40}")


if __name__ == '__main__':
    run_demo()

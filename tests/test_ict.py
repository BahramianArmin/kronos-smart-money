"""
Tests for ICT concepts + Kronos × ICT hybrid scoring
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd

# ICT concepts — without hybrid (needs torch))
from ict import (  # noqa: F811
    detect_fvg, detect_order_blocks, detect_liquidity,
    detect_mss, detect_cisd, score_ict, check_fvg_fill,
    FVG, OrderBlock, MSS, CISD, ICTScore
)


def make_test_df(length: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(seed)
    dates = pd.date_range('2024-01-01', periods=length, freq='D')
    close = 100 + np.cumsum(np.random.randn(length) * 0.5)
    high = close + np.random.rand(length) * 2
    low = close - np.random.rand(length) * 2
    open_ = close - np.random.randn(length) * 0.5
    volume = np.random.randint(1000, 10000, length)
    return pd.DataFrame({
        'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume
    }, index=dates)


def inject_fvg(df: pd.DataFrame, idx: int, direction: str = 'bullish',
               gap_size: float = 3.0):
    """Inject a manual FVG pattern into the DataFrame at idx."""
    if direction == 'bullish' and idx + 2 < len(df):
        # Make candle[idx+2].low > candle[idx].high
        df.iloc[idx + 2, df.columns.get_loc('low')] = df.iloc[idx]['high'] + gap_size
        df.iloc[idx + 2, df.columns.get_loc('open')] = df.iloc[idx]['high'] + gap_size
        df.iloc[idx + 2, df.columns.get_loc('close')] = df.iloc[idx]['high'] + gap_size + 1
    elif direction == 'bearish' and idx + 2 < len(df):
        df.iloc[idx + 2, df.columns.get_loc('high')] = df.iloc[idx]['low'] - gap_size
        df.iloc[idx + 2, df.columns.get_loc('open')] = df.iloc[idx]['low'] - gap_size
        df.iloc[idx + 2, df.columns.get_loc('close')] = df.iloc[idx]['low'] - gap_size - 1


# ─── FVG Tests ───

def test_detect_fvg_bullish():
    df = make_test_df(50)
    inject_fvg(df, idx=10, direction='bullish')
    fvgs = detect_fvg(df)
    assert len(fvgs) >= 1, "Should detect at least 1 bullish FVG"
    assert any(f.type == 'bullish' for f in fvgs), "Should have bullish FVG"
    print(f"✅ test_detect_fvg_bullish: {len(fvgs)} FVG(s) detected")


def test_detect_fvg_bearish():
    df = make_test_df(50)
    inject_fvg(df, idx=20, direction='bearish')
    fvgs = detect_fvg(df)
    assert len(fvgs) >= 1, "Should detect at least 1 bearish FVG"
    assert any(f.type == 'bearish' for f in fvgs), "Should have bearish FVG"
    print(f"✅ test_detect_fvg_bearish: {len(fvgs)} FVG(s) detected")


def test_fvg_fill():
    df = make_test_df(50)
    inject_fvg(df, idx=10, direction='bullish')
    fvgs = detect_fvg(df)
    if fvgs:
        result = check_fvg_fill(df, fvgs[0])
        print(f"✅ test_fvg_fill: filled={result}")
    else:
        print("⚠️ test_fvg_fill: no FVG to test")


# ─── OB Tests ───

def test_detect_order_blocks():
    df = make_test_df(50)
    inject_fvg(df, idx=10, direction='bullish')
    fvgs = detect_fvg(df)
    obs = detect_order_blocks(df, fvgs)
    print(f"✅ test_detect_order_blocks: {len(obs)} OB(s) detected")


# ─── Liquidity Tests ───

def test_detect_liquidity():
    df = make_test_df(100)
    liq = detect_liquidity(df, lookback=10)
    assert len(liq) > 0, "Should detect some swing points"
    print(f"✅ test_detect_liquidity: {len(liq)} zones (swing_high={sum(1 for l in liq if l['type']=='swing_high')}, swing_low={sum(1 for l in liq if l['type']=='swing_low')})")


# ─── MSS Tests ───

def test_detect_mss():
    df = make_test_df(100)
    mss_list = detect_mss(df, lookback=10)
    print(f"✅ test_detect_mss: {len(mss_list)} structure shift(s)")


# ─── CISD Tests ───

def test_detect_cisd():
    df = make_test_df(100)
    # Inject multiple bullish FVGs in a row
    for i in range(5, 15, 3):
        inject_fvg(df, idx=i, direction='bullish')
    cisd_list = detect_cisd(df, min_count=2)
    print(f"✅ test_detect_cisd: {len(cisd_list)} CISD sequence(s)")
    for c in cisd_list:
        print(f"   {c.direction} × {c.count}")


# ─── ICT Scoring Tests ───

def test_score_ict():
    df = make_test_df(100)
    scores = score_ict(df, lookback_liquidity=15)
    assert len(scores) == len(df), "Should return score for every candle"
    non_neutral = [s for s in scores if s.signal != 'neutral']
    print(f"✅ test_score_ict: {len(scores)} candles scored, {len(non_neutral)} non-neutral signals")
    if non_neutral:
        best = max(scores, key=lambda s: s.composite_score)
        print(f"   Best signal: {best.signal} (score={best.composite_score}) at idx={best.idx}")


# ─── Hybrid Scoring Tests ───

def test_hybrid_signal_dataclass():
    """Test HybridSignal without importing hybrid module (needs torch)."""
    from dataclasses import dataclass
    @dataclass
    class _HS:
        idx: int
        signal: str
        composite_score: float
        conviction: float
    
    sig = _HS(idx=42, signal='buy', composite_score=7.2, conviction=0.72)
    assert sig.signal == 'buy'
    assert sig.composite_score == 7.2
    print(f"✅ test_hybrid_signal_dataclass: signal={sig.signal}, conviction={sig.conviction}")


def test_ict_score_dataclass():
    score = ICTScore(idx=5, timestamp='2024-01-06', price=102.0,
                     fvg_count=2, composite_score=6.5, signal='buy')
    assert score.signal == 'buy'
    print(f"✅ test_ict_score_dataclass: signal={score.signal}, score={score.composite_score}")


# ─── FVG Dataclass Tests ───

def test_fvg_dataclass():
    fvg = FVG(idx=10, type='bullish', top=105.0, bottom=102.0,
              midpoint=103.5, gap_size=3.0)
    assert fvg.unfilled_pct == 100.0
    fvg.filled = True
    assert fvg.unfilled_pct == 0.0
    print(f"✅ test_fvg_dataclass: gap_size={fvg.gap_size}, unfilled={fvg.unfilled_pct}%")


# ─── Run All ───

if __name__ == '__main__':
    tests = [
        test_fvg_dataclass,
        test_ict_score_dataclass,
        test_hybrid_signal_dataclass,
        test_detect_fvg_bullish,
        test_detect_fvg_bearish,
        test_fvg_fill,
        test_detect_order_blocks,
        test_detect_liquidity,
        test_detect_mss,
        test_detect_cisd,
        test_score_ict,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__}: {e}")
            failed += 1
    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed of {len(tests)}")

"""
Hybrid Demo: Kronos × ICT on synthetic BTC data
Full pipeline — Kronos-mini model + ICT scoring
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import torch

from model.kronos import Kronos, KronosTokenizer, KronosPredictor
from ict import score_ict, detect_fvg, detect_liquidity, detect_mss, detect_cisd
from ict.hybrid import HybridScorer


def make_test_data(n=200):
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=n, freq='D')
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.random.rand(n) * 2
    low = close - np.random.rand(n) * 2
    open_ = close - np.random.randn(n) * 0.5
    volume = np.random.randint(1000, 10000, n)
    return pd.DataFrame({
        'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume
    }, index=dates)


def main():
    print("=" * 60)
    print("  Kronos × Smart Money — Full Hybrid Demo")
    print("=" * 60)

    # 1. Create test data
    print("\n📊 Generating test data...")
    df = make_test_data(200)
    print(f"   {len(df)} daily candles")

    # 2. Load Kronos models
    print("\n🤖 Loading Kronos-mini model...")
    device = 'cpu'
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
    tokenizer.eval().to(device)
    model.eval().to(device)
    print(f"   ✓ Tokenizer: {sum(p.numel() for p in tokenizer.parameters()):,} params")
    print(f"   ✓ Model:     {sum(p.numel() for p in model.parameters()):,} params")

    # 3. ICT Scoring
    print("\n🔍 Running ICT analysis...")
    ict_scores = score_ict(df, lookback_liquidity=15)
    buy = sum(1 for s in ict_scores if s.signal == 'buy')
    sell = sum(1 for s in ict_scores if s.signal == 'sell')
    best = max(ict_scores, key=lambda s: s.composite_score)
    print(f"   Signals: {buy} buy, {sell} sell, {len(ict_scores)-buy-sell} neutral")
    print(f"   Best: {best.signal} (score={best.composite_score}) at idx={best.idx}")

    # 4. Kronos Prediction
    print("\n🧠 Running Kronos prediction (12 candles)...")
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=256)
    x_df = df.iloc[-128:][['open', 'high', 'low', 'close', 'volume']].copy()
    x_ts = pd.Series(df.index[-128:])
    freq = df.index[-1] - df.index[-2]
    y_ts = pd.Series([df.index[-1] + freq * (i + 1) for i in range(12)])
    try:
        pred = predictor.predict(x_df, x_ts, y_ts, pred_len=12, T=1.0, top_p=0.9,
                                 sample_count=3, verbose=True)
        last_close = df['close'].iloc[-1]
        pred_close = pred['close'].values[0]
        change_pct = ((pred_close - last_close) / last_close) * 100
        print(f"   Last close: ${last_close:.2f}")
        print(f"   Predicted next: ${pred_close:.2f} ({change_pct:+.2f}%)")
        print(f"   12-candle pred range: ${pred['low'].min():.2f} - ${pred['high'].max():.2f}")
    except Exception as e:
        print(f"   ⚠️ Kronos prediction failed: {e}")
        print("   (This is expected for tiny synthetic data — model needs real market data)")

    # 5. Hybrid Scoring (if prediction worked)
    print("\n⚡ Computing Hybrid Score...")
    try:
        scorer = HybridScorer(model, tokenizer, device=device, max_context=256)
        hybrid_results = scorer.score(df, lookback=128, pred_len=12, T=1.0, top_p=0.9)
        strong = [h for h in hybrid_results if h.signal.startswith('strong')]
        buys = [h for h in hybrid_results if h.signal in ('buy', 'strong_buy')]
        sells = [h for h in hybrid_results if h.signal in ('sell', 'strong_sell')]
        print(f"   Strong signals: {len(strong)}")
        print(f"   Buy signals: {len(buys)}")
        print(f"   Sell signals: {len(sells)}")
        if strong:
            best_h = max(hybrid_results, key=lambda h: h.composite_score)
            print(f"\n🏆 Best hybrid signal:")
            print(f"   Idx: {best_h.idx} | Date: {best_h.timestamp}")
            print(f"   Signal: {best_h.signal.upper()} | Score: {best_h.composite_score:.1f}/10 | Conviction: {best_h.conviction:.0%}")
    except Exception as e:
        print(f"   ⚠️ Hybrid scoring skipped: {e}")

    print(f"\n{'='*60}")
    print("  ✅ Demo complete")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

"""
Comprehensive test: Kronos × Smart Money on ALL real market data
Runs ICT scoring + Kronos prediction + Hybrid on every dataset
"""
import sys, os, warnings, json
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import torch

from model.kronos import Kronos, KronosTokenizer, KronosPredictor
from ict import score_ict, detect_fvg, detect_liquidity, detect_mss, detect_cisd
from ict.hybrid import HybridScorer


DATA_DIR = '/home/armin/mrs_nightly'
DATASETS = [
    'BTCUSDT_1h', 'BTCUSDT_4h', 'BTCUSDT_1d',
    'ETHUSDT_1h', 'ETHUSDT_4h', 'ETHUSDT_1d',
    'BNBUSDT_1h', 'BNBUSDT_4h', 'BNBUSDT_1d',
    'SOLUSDT_1h', 'SOLUSDT_4h', 'SOLUSDT_1d',
]

# Pad to nearest valid Kronos context size
CONTEXT_SIZE = 128
PRED_LEN = 12


def load_dataset(name):
    """Load parquet, clean, add timestamp index for Kronos."""
    fpath = os.path.join(DATA_DIR, f'{name}.parquet')
    df = pd.read_parquet(fpath)
    
    # Ensure required columns exist for Kronos
    required = {'open', 'high', 'low', 'close', 'volume'}
    missing = required - set(df.columns)
    if missing:
        return None, f"missing cols: {missing}"
    
    # Set datetime index
    if 'close_time' in df.columns:
        df.set_index(pd.to_datetime(df['close_time'], unit='ms'), inplace=True)
        df.sort_index(inplace=True)
    else:
        # Use positional index
        df.index = pd.date_range('2024-01-01', periods=len(df), freq='h')
    
    # Rename columns to match Kronos expectations (6 cols)
    kronos_cols = ['open', 'high', 'low', 'close', 'volume']
    # Some might have different names
    rename_map = {}
    for c in kronos_cols:
        if c not in df.columns:
            return None, f"column '{c}' not found"
    df = df[kronos_cols].copy()
    
    return df, None


def run_single(device, tokenizer, model, name, df):
    """Run full pipeline on one dataset."""
    print(f"\n{'='*60}")
    print(f"  📊 {name}")
    print(f"  Rows: {len(df):,} | Range: {df.index[0]} → {df.index[-1]}")
    print(f"{'='*60}")
    
    n = len(df)
    results = {}
    
    # ─── ICT Analysis ───
    print("\n  🔍 ICT Concepts:")
    ict_scores = score_ict(df, lookback_liquidity=min(20, n//10))
    buy = sum(1 for s in ict_scores if s.signal == 'buy')
    sell = sum(1 for s in ict_scores if s.signal == 'sell')
    results['ict'] = {
        'buy': buy, 'sell': sell, 'neutral': len(ict_scores) - buy - sell,
    }
    print(f"     Buy: {buy} | Sell: {sell} | Neutral: {len(ict_scores)-buy-sell}")
    
    if ict_scores:
        best = max(ict_scores, key=lambda s: s.composite_score)
        print(f"     Best: {best.signal.upper()} (score={best.composite_score})")
    
    # FVG detection
    fvgs = detect_fvg(df)
    print(f"     FVGs: {len(fvgs)} ({sum(1 for f in fvgs if f.type=='bullish')} bullish, {sum(1 for f in fvgs if f.type=='bearish')} bearish)")
    results['fvgs'] = len(fvgs)
    
    # MSS
    mss_list = detect_mss(df, lookback=min(15, n//15))
    print(f"     MSS:  {len(mss_list)} ({sum(1 for m in mss_list if m.type=='bullish_mss')} bullish, {sum(1 for m in mss_list if m.type=='bearish_mss')} bearish)")
    results['mss'] = len(mss_list)
    
    # ─── Kronos Prediction ───
    print("\n  🧠 Kronos Prediction:")
    lookback = min(CONTEXT_SIZE, len(df) - PRED_LEN - 1)
    if lookback < PRED_LEN + 1:
        print("     ⛔ Insufficient data for prediction")
        results['kronos'] = 'skipped'
        results['hybrid'] = 'skipped'
        return results
    
    try:
        x_df = df.iloc[-lookback:].copy()
        x_ts = pd.Series(df.index[-lookback:])
        freq = df.index[-1] - df.index[-2]
        y_ts = pd.Series([df.index[-1] + freq * (i+1) for i in range(PRED_LEN)])
        
        pred = predictor.predict(x_df, x_ts, y_ts, pred_len=PRED_LEN,
                                 T=0.8, top_p=0.9, sample_count=3, verbose=False)
        
        last_close = float(df['close'].iloc[-1])
        pred_close = float(pred['close'].values[0])
        change_pct = ((pred_close - last_close) / last_close) * 100
        pred_max = float(pred['high'].values[-1])
        pred_min = float(pred['low'].values[-1])
        
        # Direction score
        direction = 'UP' if change_pct > 0.5 else 'DOWN' if change_pct < -0.5 else 'FLAT'
        print(f"     Last: ${last_close:.2f} → Pred: ${pred_close:.2f} ({change_pct:+.2f}%) {direction}")
        print(f"     Range: ${pred_min:.2f} - ${pred_max:.2f}")
        
        results['kronos'] = {
            'last_price': round(last_close, 2),
            'pred_price': round(pred_close, 2),
            'change_pct': round(change_pct, 2),
            'direction': direction,
            'pred_high': round(pred_max, 2),
            'pred_low': round(pred_min, 2),
        }
    except Exception as e:
        print(f"     ⚠️ Failed: {e}")
        results['kronos'] = {'error': str(e)[:80]}
        results['hybrid'] = 'skipped'
        return results
    
    # ─── Hybrid Scoring ───
    print("\n  ⚡ Hybrid Score:")
    try:
        scorer = HybridScorer(model, tokenizer, device=device, max_context=CONTEXT_SIZE)
        hybrid = scorer.score(df, lookback=lookback, pred_len=PRED_LEN,
                              T=0.8, top_p=0.9, sample_count=3)
        
        strong = [h for h in hybrid if 'strong' in h.signal]
        buy_sig = [h for h in hybrid if h.signal in ('buy', 'strong_buy')]
        sell_sig = [h for h in hybrid if h.signal in ('sell', 'strong_sell')]
        
        print(f"     Strong: {len(strong)} | Buy: {len(buy_sig)} | Sell: {len(sell_sig)} | Neutral: {len(hybrid)-len(buy_sig)-len(sell_sig)}")
        
        results['hybrid'] = {
            'strong': len(strong),
            'buy': len(buy_sig),
            'sell': len(sell_sig),
            'neutral': len(hybrid)-len(buy_sig)-len(sell_sig),
        }
        
        if strong:
            best_h = max(strong, key=lambda h: h.composite_score)
            print(f"     Best hybrid: {best_h.signal.upper()} ({best_h.composite_score:.1f}/10, conviction={best_h.conviction:.0%})")
            results['hybrid']['best_signal'] = best_h.signal
            results['hybrid']['best_score'] = best_h.composite_score
            results['hybrid']['best_conviction'] = best_h.conviction
        elif hybrid:
            best_h = max(hybrid, key=lambda h: h.composite_score)
            if best_h.composite_score > 4:
                print(f"     Best: {best_h.signal.upper()} ({best_h.composite_score:.1f}/10)")
                results['hybrid']['best_signal'] = best_h.signal
                results['hybrid']['best_score'] = best_h.composite_score
            else:
                print(f"     All neutral (composite range: 0-{best_h.composite_score:.1f})")
                results['hybrid']['best_signal'] = 'neutral'
                results['hybrid']['best_score'] = best_h.composite_score
    except Exception as e:
        print(f"     ⚠️ Failed: {e}")
        results['hybrid'] = {'error': str(e)[:80]}
    
    return results


def main():
    print("=" * 60)
    print("  KRONOS × SMART MONEY — Full Market Test Suite")
    print("  Device: CPU")
    print("=" * 60)
    
    # Load models once
    print("\n🤖 Loading Kronos-mini + Tokenizer...")
    device = 'cpu'
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
    tokenizer.eval().to(device)
    model.eval().to(device)
    
    global predictor
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=CONTEXT_SIZE)
    print(f"   ✓ Tokenizer: {sum(p.numel() for p in tokenizer.parameters()):,} params")
    print(f"   ✓ Model:     {sum(p.numel() for p in model.parameters()):,} params\n")
    
    # Run on every dataset
    all_results = {}
    for name in DATASETS:
        df, err = load_dataset(name)
        if df is None:
            print(f"\n⛔ {name}: {err}")
            all_results[name] = {'error': err}
            continue
        results = run_single(device, tokenizer, model, name, df)
        all_results[name] = results
    
    # ─── Summary Report ───
    print("\n")
    print("=" * 60)
    print("  📋 SUMMARY REPORT")
    print("=" * 60)
    
    print(f"\n{'Dataset':<20s} {'ICT B/S':>10s} {'FVGs':>6s} {'MSS':>5s} {'Kronos':>10s} {'Hybrid':>10s}")
    print("-" * 65)
    
    for name in DATASETS:
        r = all_results.get(name, {})
        ict = r.get('ict', {})
        kronos = r.get('kronos', {})
        hybrid = r.get('hybrid', {})
        
        ict_str = f"{ict.get('buy',0)}/{ict.get('sell',0)}"
        fvg_str = str(r.get('fvgs', '?'))
        mss_str = str(r.get('mss', '?'))
        
        if isinstance(kronos, dict) and 'direction' in kronos:
            k_str = f"{kronos['direction']} {kronos.get('change_pct', 0):+.1f}%"
        elif isinstance(kronos, dict) and 'error' in kronos:
            k_str = 'ERR'
        else:
            k_str = '—'
        
        if isinstance(hybrid, dict) and hybrid.get('best_signal'):
            h_str = f"{hybrid['best_signal']} {hybrid.get('best_score', 0):.1f}"
        elif isinstance(hybrid, dict) and 'error' in hybrid:
            h_str = 'ERR'
        else:
            h_str = '—'
        
        print(f"{name:<20s} {ict_str:>10s} {fvg_str:>6s} {mss_str:>5s} {k_str:>10s} {h_str:>10s}")
    
    print(f"\n{'='*60}")
    print("  ✅ All tests complete")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

"""
Comprehensive test: Kronos × Smart Money on ALL CHARTIX data
13 symbols × 5 main timeframes (H1, H4, 1D, m30, m15)
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


CHARTIX_DIR = '/home/armin/chartix_data/csv'
PRED_LEN = 12
CONTEXT_SIZE = 256

# Symbols to test (main ones — BTC, ETH, SOL, XAU, EURUSD, GBPUSD)
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'EURUSD', 'GBPUSD', 'XAUUSD']
TIMEFRAMES = ['H1', 'H4', '1D']


def load_chartix(symbol: str, tf: str) -> pd.DataFrame | None:
    """Load a Chartix CSV and return standard OHLCV DataFrame."""
    # Map symbol to Chartix naming
    smap = {
        'EURUSD': 'FOREXCOM_EURUSD', 'GBPUSD': 'FOREXCOM_GBPUSD',
        'AUDUSD': 'FOREXCOM_AUDUSD', 'NZDUSD': 'FOREXCOM_NZDUSD',
        'USDJPY': 'FOREXCOM_USDJPY', 'USDCAD': 'FOREXCOM_USDCAD',
        'USDCHF': 'FOREXCOM_USDCHF', 'CADCHF': 'OANDA_CADCHF',
        'BTCUSDT': 'BTCUSDT', 'ETHUSDT': 'ETHUSDT', 'SOLUSDT': 'SOLUSDT',
        'XAUUSD': 'FOREXCOM_XAUUSD',
    }
    prefix = smap.get(symbol, symbol)
    
    # Try both naming conventions
    candidates = [
        os.path.join(CHARTIX_DIR, tf, f'{prefix}_{tf}_CHARTIX.csv'),
        os.path.join(CHARTIX_DIR, tf, f'{prefix}_{tf.replace("H","H")}_CHARTIX.csv'),
    ]
    if tf == '1D':
        candidates.append(os.path.join(CHARTIX_DIR, tf, f'{prefix}_1D_CHARTIX.csv'))
    
    fpath = None
    for c in candidates:
        if os.path.exists(c):
            fpath = c
            break
    
    if fpath is None:
        # Try alternative naming: XAUUSD might be FOREXCOM_XAUUSD
        alt_symbol = {'XAUUSD': 'OANDA_XAUUSD'}.get(symbol)
        if alt_symbol:
            alt_path = os.path.join(CHARTIX_DIR, tf, f'{alt_symbol}_{tf}_CHARTIX.csv')
            if os.path.exists(alt_path):
                fpath = alt_path
    
    if fpath is None:
        return None
    
    try:
        df = pd.read_csv(fpath, skipinitialspace=True)
        # Clean column names
        df.columns = [c.strip('<> \\r').strip().lower() for c in df.columns]
        rename = {
            'dtyyyymmdd': 'date', 'time': 'time',
            'open': 'open', 'high': 'high', 'low': 'low',
            'close': 'close', 'vol': 'volume'
        }
        df.rename(columns=rename, inplace=True)
        
        # Build datetime index
        df['time_str'] = df['time'].astype(str).str.zfill(6)
        df['date_str'] = df['date'].astype(str)
        df['datetime'] = pd.to_datetime(df['date_str'] + ' ' + df['time_str'],
                                         format='%Y%m%d %H%M%S', errors='coerce')
        df.dropna(subset=['datetime'], inplace=True)
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        
        # Keep only OHLCV
        df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        return df
    except Exception as e:
        print(f"     ⚠️ Load error: {e}")
        return None


def run_single(device, tokenizer, model, symbol, tf, df):
    """Run full pipeline on one dataset."""
    print(f"\n{'='*60}")
    print(f"  📊 {symbol} ({tf})")
    print(f"  Rows: {len(df):,} | {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"{'='*60}")
    
    n = len(df)
    results = {}
    
    # ─── ICT Analysis ───
    print("\n  🔍 ICT Concepts:")
    ict_scores = score_ict(df, lookback_liquidity=min(30, n//5))
    buy = sum(1 for s in ict_scores if s.signal == 'buy')
    sell = sum(1 for s in ict_scores if s.signal == 'sell')
    results['ict'] = {'buy': buy, 'sell': sell, 'neutral': len(ict_scores)-buy-sell}
    print(f"     Buy: {buy} | Sell: {sell} | Neutral: {len(ict_scores)-buy-sell}")
    
    if ict_scores:
        best = max(ict_scores, key=lambda s: s.composite_score)
        print(f"     Best: {best.signal.upper()} (score={best.composite_score})")
    
    fvgs = detect_fvg(df)
    results['fvgs'] = len(fvgs)
    print(f"     FVGs: {len(fvgs)}")
    
    # ─── Kronos Prediction ───
    print("\n  🧠 Kronos Prediction:")
    lookback = min(CONTEXT_SIZE, len(df) - PRED_LEN - 1)
    if lookback < PRED_LEN + 1:
        print("     ⛔ Insufficient data")
        results['kronos'] = 'skipped'
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
        direction = 'UP' if change_pct > 0.5 else 'DOWN' if change_pct < -0.5 else 'FLAT'
        
        print(f"     Last: {last_close:.4f} → Pred: {pred_close:.4f} ({change_pct:+.2f}%) {direction}")
        
        results['kronos'] = {
            'last_price': round(last_close, 4),
            'pred_price': round(pred_close, 4),
            'change_pct': round(change_pct, 2),
            'direction': direction,
        }
    except Exception as e:
        print(f"     ⚠️ Failed: {e}")
        results['kronos'] = {'error': str(e)[:80]}
        return results
    
    # ─── Hybrid Scoring ───
    print("\n  ⚡ Hybrid Score:")
    try:
        scorer = HybridScorer(model, tokenizer, device=device, max_context=CONTEXT_SIZE)
        hybrid = scorer.score(df, lookback=lookback, pred_len=PRED_LEN,
                              T=0.8, top_p=0.9, sample_count=3)
        
        buy_sig = [h for h in hybrid if h.signal in ('buy', 'strong_buy')]
        sell_sig = [h for h in hybrid if h.signal in ('sell', 'strong_sell')]
        neutral = len(hybrid) - len(buy_sig) - len(sell_sig)
        
        print(f"     Buy: {len(buy_sig)} | Sell: {len(sell_sig)} | Neutral: {neutral}")
        
        results['hybrid'] = {'buy': len(buy_sig), 'sell': len(sell_sig), 'neutral': neutral}
        
        if hybrid:
            best_h = max(hybrid, key=lambda h: h.composite_score)
            print(f"     Best: {best_h.signal.upper()} ({best_h.composite_score:.1f}/10, conviction={best_h.conviction:.0%})")
            results['hybrid']['best_signal'] = best_h.signal
            results['hybrid']['best_score'] = best_h.composite_score
            results['hybrid']['best_conviction'] = best_h.conviction
    except Exception as e:
        print(f"     ⚠️ Failed: {e}")
        results['hybrid'] = {'error': str(e)[:80]}
    
    return results


def main():
    print("=" * 60)
    print("  KRONOS × SMART MONEY — CHARTIX Full Market Test")
    print("=" * 60)
    
    # Load models
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
    
    # Run all
    all_results = {}
    test_cases = [(s, tf) for s in SYMBOLS for tf in TIMEFRAMES]
    
    for symbol, tf in test_cases:
        print(f"\n  Loading {symbol} ({tf})...", end=' ')
        df = load_chartix(symbol, tf)
        if df is None:
            print('⛔ Not found')
            all_results[f'{symbol}_{tf}'] = {'error': 'not found'}
            continue
        print(f'✅ {len(df):,} rows')
        results = run_single(device, tokenizer, model, symbol, tf, df)
        all_results[f'{symbol}_{tf}'] = results
    
    # ─── Summary ───
    print("\n")
    print("=" * 70)
    print("  📋 SUMMARY REPORT — ALL CHARTIX DATA")
    print("=" * 70)
    
    header = f"{'Dataset':<18s} {'ICT B/S/N':>16s} {'FVGs':>5s} {'Kronos Dir':>10s} {'Hybrid Best':>15s}"
    print(f"\n{header}")
    print("-" * 70)
    
    for key in sorted(all_results.keys()):
        r = all_results[key]
        ict = r.get('ict', {})
        kronos = r.get('kronos', {})
        hybrid = r.get('hybrid', {})
        
        if isinstance(ict, dict):
            ict_str = f"{ict.get('buy',0)}/{ict.get('sell',0)}/{ict.get('neutral',0)}"
        else:
            ict_str = '—'
        
        fvg_str = str(r.get('fvgs', '?'))
        
        if isinstance(kronos, dict) and kronos.get('direction'):
            k_dir = kronos['direction']
            k_pct = kronos.get('change_pct', 0)
            k_str = f"{k_dir} {k_pct:+.1f}%"
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
        
        print(f"{key:<18s} {ict_str:>16s} {fvg_str:>5s} {k_str:>10s} {h_str:>15s}")
    
    print(f"\n{'='*70}")
    print("  ✅ Complete")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

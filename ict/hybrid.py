"""
Kronos × ICT — Hybrid Scoring Engine
Combines Kronos AI price predictions with ICT concept signals
"""

import numpy as np
import pandas as pd
from typing import Optional, List
from dataclasses import dataclass, field

from model.kronos import Kronos, KronosTokenizer, KronosPredictor
from ict import score_ict, ICTScore, detect_fvg, check_fvg_fill


@dataclass
class HybridSignal:
    """Combined signal from Kronos AI + ICT concepts"""
    idx: int
    timestamp: str
    price: float
    # Kronos components
    kronos_pred_next: Optional[float] = None    # predicted close next period
    kronos_pred_high: Optional[float] = None    # predicted high
    kronos_pred_low: Optional[float] = None     # predicted low
    kronos_direction: float = 0.0               # +1 = up, -1 = down, 0 = flat
    kronos_confidence: float = 0.0              # 0-1 based on entropy
    kronos_score: float = 0.0                   # 0-10
    # ICT components
    ict: Optional[ICTScore] = None
    # Combined
    composite_score: float = 0.0
    signal: str = 'neutral'     # 'strong_buy', 'buy', 'neutral', 'sell', 'strong_sell'
    conviction: float = 0.0     # 0-1


class HybridScorer:
    """
    Combines Kronos price prediction with ICT concepts.
    
    Strategy:
    1. Kronos predicts next N candles (OHLCV)
    2. ICT detects FVG, OB, Liquidity, MSS on current data
    3. Hybrid score: Kronos direction + ICT score + alignment check
    4. Trade signal if both agree AND conviction is high
    """

    def __init__(self, kronos_model: Kronos, tokenizer: KronosTokenizer,
                 device: str = 'cpu', max_context: int = 512):
        self.predictor = KronosPredictor(
            kronos_model, tokenizer,
            device=device, max_context=max_context
        )
        self.max_context = max_context

    def score(self, df: pd.DataFrame, lookback: int = 200,
              pred_len: int = 12, T: float = 1.0, top_p: float = 0.9,
              sample_count: int = 3) -> List[HybridSignal]:
        """
        Compute hybrid signals for the entire DataFrame.
        
        Args:
            df: OHLCV DataFrame with datetime index
            lookback: recent candles to use for Kronos prediction
            pred_len: candles to predict forward
            T: sampling temperature (lower = more conservative)
            top_p: nucleus sampling threshold
            sample_count: samples to average over
        
        Returns:
            List of HybridSignal for each candle
        """
        n = len(df)
        if n < lookback:
            lookback = n - 1

        # 1. Compute ICT scores for all candles
        ict_scores = score_ict(df)
        ict_by_idx = {s.idx: s for s in ict_scores}

        # 2. Get Kronos prediction for the most recent window
        x_df = df.iloc[-lookback:][['open', 'high', 'low', 'close', 'volume']].copy()
        x_timestamp = pd.Series(df.index[-lookback:])
        
        # Create future timestamps (assume same frequency)
        if len(df.index) > 1:
            freq = df.index[-1] - df.index[-2]
        else:
            freq = pd.Timedelta(days=1)
        y_timestamp = pd.Series(
            [df.index[-1] + freq * (i + 1) for i in range(pred_len)]
        )

        # Generate prediction
        try:
            pred_df = self.predictor.predict(
                x_df, x_timestamp, y_timestamp,
                pred_len=pred_len, T=T, top_p=top_p,
                sample_count=sample_count, verbose=False
            )
            # Store prediction values
            pred_close = pred_df['close'].values
            pred_high = pred_df['high'].values
            pred_low = pred_df['low'].values
            has_prediction = True
        except Exception as e:
            print(f"[Kronos] Prediction failed: {e}")
            has_prediction = False

        # 3. Build hybrid signals
        results = []
        for i in range(n):
            row = df.iloc[i]
            ict_i = ict_by_idx.get(i)
            
            sig = HybridSignal(
                idx=i,
                timestamp=str(df.index[i]) if hasattr(df.index[i], 'strftime') else str(df.index[i]),
                price=float(row['close']),
                ict=ict_i,
            )

            if has_prediction and i == n - 1:
                # Kronos prediction for current candle (nearest future)
                sig.kronos_pred_next = float(pred_close[0])
                sig.kronos_pred_high = float(pred_high[0])
                sig.kronos_pred_low = float(pred_low[0])
                
                # Direction score (-1 to +1)
                current_close = float(row['close'])
                if current_close > 0:
                    change_pct = (pred_close[0] - current_close) / current_close * 100
                    sig.kronos_direction = np.clip(change_pct / 5.0, -1.0, 1.0)

            # ─── Kronos Score ───
            # Based on prediction strength and recent volatility
            if has_prediction:
                pred_vol = np.std(pred_close[:min(6, len(pred_close))])
                hist_vol = np.std(df['close'].iloc[max(0, i-20):i+1])
                
                if hist_vol > 0:
                    vol_ratio = pred_vol / hist_vol
                    # Higher vol ratio = more confident directional move
                    confidence = min(abs(vol_ratio), 1.0)
                    if abs(sig.kronos_direction) > 0.3:
                        sig.kronos_confidence = confidence
                    else:
                        sig.kronos_confidence = 0.0

                sig.kronos_score = abs(sig.kronos_direction) * sig.kronos_confidence * 10

            # ─── Composite Score ───
            if ict_i:
                # Weight: 60% ICT, 40% Kronos
                ict_part = ict_i.composite_score * 0.6
                kronos_part = min(sig.kronos_score, 10) * 0.4
                
                # Alignment bonus: if both agree on direction
                kronos_direction_sign = np.sign(sig.kronos_direction)
                if ict_i.signal == 'buy' and kronos_direction_sign > 0.3:
                    sig.composite_score = min(ict_part + kronos_part + 1.5, 10)
                elif ict_i.signal == 'sell' and kronos_direction_sign < -0.3:
                    sig.composite_score = min(ict_part + kronos_part + 1.5, 10)
                else:
                    sig.composite_score = min(ict_part + kronos_part, 10)
            else:
                sig.composite_score = min(sig.kronos_score, 10)

            # ─── Signal Generation ───
            if sig.composite_score >= 8.0:
                sig.signal = 'strong_buy' if (ict_i and ict_i.signal == 'buy') or sig.kronos_direction > 0.3 \
                             else 'strong_sell' if (ict_i and ict_i.signal == 'sell') or sig.kronos_direction < -0.3 \
                             else 'neutral'
            elif sig.composite_score >= 6.0:
                sig.signal = 'buy' if (ict_i and ict_i.signal in ('buy',)) or sig.kronos_direction > 0.5 \
                             else 'sell'
            else:
                sig.signal = 'neutral'

            # Conviction
            sig.conviction = round(sig.composite_score / 10.0, 2)

            results.append(sig)

        return results

    def predict_next(self, df: pd.DataFrame, pred_len: int = 12,
                     T: float = 1.0, top_p: float = 0.9,
                     sample_count: int = 3) -> pd.DataFrame:
        """
        Quick wrapper: predict only the next N candles.
        Returns DataFrame with predicted OHLCV.
        """
        lookback = min(self.max_context, len(df) - 1)
        x_df = df.iloc[-lookback:][['open', 'high', 'low', 'close', 'volume']].copy()
        x_timestamp = pd.Series(df.index[-lookback:])

        if len(df.index) > 1:
            freq = df.index[-1] - df.index[-2]
        else:
            freq = pd.Timedelta(days=1)
        y_timestamp = pd.Series(
            [df.index[-1] + freq * (i + 1) for i in range(pred_len)]
        )

        return self.predictor.predict(
            x_df, x_timestamp, y_timestamp,
            pred_len=pred_len, T=T, top_p=top_p,
            sample_count=sample_count, verbose=False
        )

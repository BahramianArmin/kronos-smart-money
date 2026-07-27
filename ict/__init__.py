"""
ICT Concepts — Smart Money trading logic
FVG, Order Block, Liquidity, MSS/BOS, CISD
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# ─── Data Structures ───

@dataclass
class FVG:
    """Fair Value Gap — three-candle imbalance"""
    idx: int
    type: str            # 'bullish' or 'bearish'
    top: float           # upper boundary
    bottom: float        # lower boundary
    midpoint: float
    gap_size: float
    filled: bool = False
    fill_idx: Optional[int] = None

    @property
    def unfilled_pct(self) -> float:
        return 0.0 if self.filled else 100.0


@dataclass
class OrderBlock:
    """ICT Order Block — last opposing candle before move"""
    idx: int
    type: str
    top: float
    bottom: float
    strength: int = 1     # 1-3 based on context
    broken: bool = False


@dataclass
class MSS:
    """Market Structure Shift / Break of Structure"""
    idx: int
    type: str            # 'bullish_mss' or 'bearish_mss'
    level: float
    # Swing high/low that was broken
    broken_swing_idx: int
    broken_swing_level: float


@dataclass
class CISD:
    """CISD — Consecutive Imbalances in Same Direction"""
    direction: str       # 'buy' or 'sell'
    count: int
    start_idx: int
    end_idx: int
    total_imbalance: float


# ─── Detectors ───

def detect_fvg(df: pd.DataFrame) -> list[FVG]:
    """
    Detect Fair Value Gaps from 3-candle formations.
    Bullish FVG: low[i+2] > high[i]   (gap up)
    Bearish FVG: high[i+2] < low[i]   (gap down)
    """
    fvgs = []
    n = len(df)
    for i in range(n - 2):
        h_i = df.iloc[i]['high']
        l_i = df.iloc[i]['low']
        h_i2 = df.iloc[i + 2]['high']
        l_i2 = df.iloc[i + 2]['low']

        # Bullish FVG
        if l_i2 > h_i:
            fvgs.append(FVG(
                idx=i + 2,
                type='bullish',
                top=l_i2,
                bottom=h_i,
                midpoint=(l_i2 + h_i) / 2,
                gap_size=l_i2 - h_i,
            ))
        # Bearish FVG
        elif h_i2 < l_i:
            fvgs.append(FVG(
                idx=i,
                type='bearish',
                top=l_i,
                bottom=h_i2,
                midpoint=(l_i + h_i2) / 2,
                gap_size=l_i - h_i2,
            ))
    return fvgs


def check_fvg_fill(df: pd.DataFrame, fvg: FVG) -> bool:
    """Check if a FVG has been filled by subsequent price action."""
    if fvg.filled:
        return True
    for i in range(fvg.idx + 1, len(df)):
        if fvg.type == 'bullish':
            if df.iloc[i]['low'] <= fvg.top:
                fvg.filled = True
                fvg.fill_idx = i
                return True
        else:
            if df.iloc[i]['high'] >= fvg.bottom:
                fvg.filled = True
                fvg.fill_idx = i
                return True
    return False


def detect_order_blocks(df: pd.DataFrame, fvgs: list[FVG] = None) -> list[OrderBlock]:
    """
    Detect ICT Order Blocks.
    Bullish OB: last bearish candle before a bullish displacement.
    Bearish OB: last bullish candle before a bearish displacement.
    """
    obs = []
    n = len(df)

    # If FVGs provided, use them to find displacements
    if fvgs:
        for fvg in fvgs:
            # Candle right before the FVG start
            ob_idx = fvg.idx - 2  # Candle i of the 3-candle FVG pattern
            if ob_idx >= 0:
                if fvg.type == 'bullish':
                    # Looking for bearish candle before bullish gap
                    if df.iloc[ob_idx]['close'] < df.iloc[ob_idx]['open']:
                        obs.append(OrderBlock(
                            idx=ob_idx,
                            type='bullish',
                            top=df.iloc[ob_idx]['high'],
                            bottom=df.iloc[ob_idx]['low'],
                        ))
                else:
                    # Looking for bullish candle before bearish gap
                    if df.iloc[ob_idx]['close'] > df.iloc[ob_idx]['open']:
                        obs.append(OrderBlock(
                            idx=ob_idx,
                            type='bearish',
                            top=df.iloc[ob_idx]['high'],
                            bottom=df.iloc[ob_idx]['low'],
                        ))
    return obs


def detect_liquidity(df: pd.DataFrame, lookback: int = 20) -> list[dict]:
    """
    Detect liquidity zones (swing highs/lows for stop hunts).
    Uses lookback window to find local extremes.
    """
    liquidity = []
    n = len(df)
    half = lookback // 2
    for i in range(half, n - half):
        high_i = df.iloc[i]['high']
        low_i = df.iloc[i]['low']

        # Swing high
        if all(high_i > df.iloc[i - j]['high'] for j in range(1, half + 1)) and \
           all(high_i >= df.iloc[i + j]['high'] for j in range(1, half + 1)):
            liquidity.append({
                'idx': i, 'type': 'swing_high', 'level': high_i,
                'strength': 2,
            })

        # Swing low
        if all(low_i < df.iloc[i - j]['low'] for j in range(1, half + 1)) and \
           all(low_i <= df.iloc[i + j]['low'] for j in range(1, half + 1)):
            liquidity.append({
                'idx': i, 'type': 'swing_low', 'level': low_i,
                'strength': 2,
            })
    return liquidity


def detect_mss(df: pd.DataFrame, lookback: int = 20) -> list[MSS]:
    """
    Detect Market Structure Shift / Break of Structure.
    Bullish MSS: price breaks above a prior swing high, then retraces below.
    Bearish MSS: price breaks below a prior swing low, then retraces above.
    """
    from itertools import combinations

    swings = detect_liquidity(df, lookback)
    mss_list = []
    n = len(df)

    swing_highs = [s for s in swings if s['type'] == 'swing_high']
    swing_lows = [s for s in swings if s['type'] == 'swing_low']

    for i in range(1, n):
        close_i = df.iloc[i]['close']
        low_i = df.iloc[i]['low']
        high_i = df.iloc[i]['high']

        # Check break of prior swing high
        for sh in reversed(swing_highs):
            if sh['idx'] >= i:
                continue
            if close_i > sh['level']:
                # BOS confirmed — now check for MSS (retracement)
                for j in range(i + 1, min(i + 5, n)):
                    if df.iloc[j]['low'] < sh['level']:
                        mss_list.append(MSS(
                            idx=j,
                            type='bearish_mss',  # price went above then came back
                            level=sh['level'],
                            broken_swing_idx=sh['idx'],
                            broken_swing_level=sh['level'],
                        ))
                        break
                break

        # Check break of prior swing low
        for sl in reversed(swing_lows):
            if sl['idx'] >= i:
                continue
            if close_i < sl['level']:
                for j in range(i + 1, min(i + 5, n)):
                    if df.iloc[j]['high'] > sl['level']:
                        mss_list.append(MSS(
                            idx=j,
                            type='bullish_mss',
                            level=sl['level'],
                            broken_swing_idx=sl['idx'],
                            broken_swing_level=sl['level'],
                        ))
                        break
                break

    return mss_list


def detect_cisd(df: pd.DataFrame, min_count: int = 3) -> list[CISD]:
    """
    CISD — Consecutive Imbalances in Same Direction.
    Counts consecutive FVG-type gaps in the same direction.
    """
    fvgs = detect_fvg(df)

    cisd_list = []
    if not fvgs:
        return cisd_list

    current_dir = fvgs[0].type
    start_idx = fvgs[0].idx
    count = 1
    total_imb = fvgs[0].gap_size

    for i in range(1, len(fvgs)):
        if fvgs[i].type == current_dir:
            count += 1
            total_imb += fvgs[i].gap_size
        else:
            if count >= min_count:
                cisd_list.append(CISD(
                    direction='buy' if current_dir == 'bullish' else 'sell',
                    count=count,
                    start_idx=start_idx,
                    end_idx=fvgs[i - 1].idx,
                    total_imbalance=total_imb,
                ))
            current_dir = fvgs[i].type
            start_idx = fvgs[i].idx
            count = 1
            total_imb = fvgs[i].gap_size

    if count >= min_count:
        cisd_list.append(CISD(
            direction='buy' if current_dir == 'bullish' else 'sell',
            count=count,
            start_idx=start_idx,
            end_idx=fvgs[-1].idx,
            total_imbalance=total_imb,
        ))

    return cisd_list


# ─── ICT Scoring ───

@dataclass
class ICTScore:
    """Combined ICT signal score for a given index"""
    idx: int
    timestamp: str
    price: float
    fvg_count: int = 0
    nearest_fvg: Optional[FVG] = None
    ob_count: int = 0
    nearest_ob: Optional[OrderBlock] = None
    liquidity_swept: bool = False
    mss_active: int = 0      # 0=none, 1=bullish, -1=bearish
    cisd_active: Optional[CISD] = None
    composite_score: float = 0.0  # 0-10
    signal: str = 'neutral'   # 'buy', 'sell', 'neutral'


def score_ict(df: pd.DataFrame, lookback_liquidity: int = 20) -> list[ICTScore]:
    """
    Compute ICT signal score for every candle in the DataFrame.
    Combines FVG, OB, Liquidity Sweep, MSS, CISD.
    """
    fvgs = detect_fvg(df)
    obs = detect_order_blocks(df, fvgs)
    liquidity = detect_liquidity(df, lookback_liquidity)
    mss_list = detect_mss(df, lookback_liquidity)
    cisd_list = detect_cisd(df)

    # Mark filled FVGs
    for fvg in fvgs:
        check_fvg_fill(df, fvg)

    n = len(df)
    results = []

    for i in range(n):
        row = df.iloc[i]
        score = ICTScore(
            idx=i,
            timestamp=str(df.index[i]) if hasattr(df.index[i], 'strftime') else str(df.index[i]),
            price=float(row['close']),
        )

        # Active FVGs near current price
        active_fvgs = []
        for fvg in fvgs:
            if fvg.idx <= i and not fvg.filled:
                dist = abs(row['close'] - fvg.midpoint) / row['close'] * 100
                if dist < 3:  # within 3% of price
                    active_fvgs.append(fvg)
        score.fvg_count = len(active_fvgs)
        if active_fvgs:
            # Nearest unfilled FVG
            score.nearest_fvg = min(active_fvgs,
                                     key=lambda f: abs(row['close'] - f.midpoint))

        # Active OBs
        active_obs = [ob for ob in obs if ob.idx <= i and not ob.broken]
        score.ob_count = len(active_obs)
        if active_obs:
            score.nearest_ob = min(active_obs,
                                    key=lambda o: abs(row['close'] - (o.top + o.bottom) / 2))

        # Liquidity sweep (price moved past a swing high/low and came back)
        for liq in liquidity:
            if liq['idx'] < i:
                if liq['type'] == 'swing_high' and row['high'] > liq['level']:
                    score.liquidity_swept = True
                elif liq['type'] == 'swing_low' and row['low'] < liq['level']:
                    score.liquidity_swept = True

        # MSS active
        for mss in mss_list:
            if mss.idx <= i:
                if mss.type == 'bullish_mss':
                    score.mss_active = 1
                elif mss.type == 'bearish_mss':
                    score.mss_active = -1

        # CISD within last 10 candles
        for c in cisd_list:
            if c.end_idx >= i - 10 and c.end_idx <= i:
                score.cisd_active = c

        # ─── Composite Score (0-10) ───
        s = 5.0  # neutral baseline

        # FVG bonus (up to +3)
        s += min(score.fvg_count * 0.8, 3.0)

        # OB bonus (up to +1.5)
        if score.ob_count > 0:
            s += min(score.ob_count * 0.5, 1.5)

        # Liquidity sweep (up to +1.5)
        if score.liquidity_swept:
            s += 1.5

        # MSS (+/- 1)
        s += score.mss_active * 1.0

        # CISD (up to +1)
        if score.cisd_active:
            if score.cisd_active.count >= 4:
                s += 1.0
            elif score.cisd_active.count >= 3:
                s += 0.5

        score.composite_score = round(s, 1)

        # Signal direction
        if s >= 7.0 and score.mss_active == 1:
            score.signal = 'buy'
        elif s >= 7.0 and score.mss_active == -1:
            score.signal = 'sell'
        elif s >= 6.0 and score.fvg_count >= 1:
            score.signal = 'buy' if any(f.type == 'bullish' for f in active_fvgs) else \
                           'sell' if any(f.type == 'bearish' for f in active_fvgs) else 'neutral'
        else:
            score.signal = 'neutral'

        results.append(score)

    return results

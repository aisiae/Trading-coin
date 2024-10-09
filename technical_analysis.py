import pandas as pd
import numpy as np

def perform_analysis(data):
    """
    주어진 데이터에 대해 기술적 분석을 수행합니다.
    """
    if data.empty:
        print("분석할 데이터가 없습니다.")
        return {}

    # 볼린저 밴드 분석
    bb_analysis = analyze_bollinger_bands(data)
    
    # 이동평균선 분석
    ma_analysis = analyze_moving_averages(data)
    
    # RSI 분석
    rsi_analysis = analyze_rsi(data)
    
    # MACD 분석
    macd_analysis = analyze_macd(data)

    # 피보나치 분석 추가
    fibonacci_analysis = analyze_fibonacci_levels(data)

    # 전체 분석 결과 종합
    analysis_result = {
        'current_price': data['close'].iloc[-1],
        'bollinger_bands': bb_analysis,
        'moving_averages': ma_analysis,
        'rsi': rsi_analysis,
        'macd': macd_analysis,
        'fibonacci': fibonacci_analysis,
        'overall_signal': get_overall_signal(bb_analysis, ma_analysis, rsi_analysis, macd_analysis, fibonacci_analysis)
    }

    return analysis_result

def analyze_bollinger_bands(data):
    upper_band = data['upper_band_20'].iloc[-1]
    lower_band = data['lower_band_20'].iloc[-1]
    current_price = data['close'].iloc[-1]
    
    if current_price > upper_band:
        return {'signal': 'Sell', 'strength': 'Strong'}
    elif current_price < lower_band:
        return {'signal': 'Buy', 'strength': 'Strong'}
    else:
        position = (current_price - lower_band) / (upper_band - lower_band)
        if position > 0.8:
            return {'signal': 'Sell', 'strength': 'Weak'}
        elif position < 0.2:
            return {'signal': 'Buy', 'strength': 'Weak'}
        else:
            return {'signal': 'Neutral', 'strength': 'Neutral'}

def analyze_moving_averages(data):
    ma4 = data['MA4'].iloc[-1]
    ma20 = data['MA20'].iloc[-1]
    ma50 = data['MA50'].iloc[-1]
    ma200 = data['MA200'].iloc[-1]
    current_price = data['close'].iloc[-1]
    
    short_term = 'Bullish' if current_price > ma4 else 'Bearish'
    medium_term = 'Bullish' if ma4 > ma20 and ma20 > ma50 else 'Bearish'
    long_term = 'Bullish' if ma50 > ma200 else 'Bearish'
    
    return {
        'short_term': short_term,
        'medium_term': medium_term,
        'long_term': long_term
    }

def analyze_rsi(data):
    rsi = data['RSI'].iloc[-1]
    if rsi > 70:
        return {'signal': 'Sell', 'strength': 'Strong'}
    elif rsi < 30:
        return {'signal': 'Buy', 'strength': 'Strong'}
    elif rsi > 60:
        return {'signal': 'Sell', 'strength': 'Weak'}
    elif rsi < 40:
        return {'signal': 'Buy', 'strength': 'Weak'}
    else:
        return {'signal': 'Neutral', 'strength': 'Neutral'}

def analyze_macd(data):
    if 'MACD' not in data.columns or 'MACD_signal' not in data.columns:
        return {'signal': 'Neutral', 'strength': 'Weak'}
    
    macd = data['MACD'].iloc[-1]
    signal = data['MACD_signal'].iloc[-1]
    
    if macd > signal:
        return {'signal': 'Buy', 'strength': 'Strong' if macd > 0 else 'Weak'}
    else:
        return {'signal': 'Sell', 'strength': 'Strong' if macd < 0 else 'Weak'}

def analyze_fibonacci_levels(data):
    """
    피보나치 되돌림 수준을 분석합니다.
    """
    high = data['high'].max()
    low = data['low'].min()
    current_price = data['close'].iloc[-1]
    
    fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
    fib_prices = [low + (high - low) * level for level in fib_levels]
    
    # 현재 가격이 어느 피보나치 수준에 있는지 확인
    for i in range(len(fib_prices) - 1):
        if fib_prices[i] <= current_price < fib_prices[i + 1]:
            lower_level = fib_levels[i]
            upper_level = fib_levels[i + 1]
            break
    else:
        return {'signal': 'Neutral', 'strength': 'Weak'}
    
    # 신호 결정
    if current_price < fib_prices[2]:  # 0.382 수준 아래
        signal = 'Buy'
        strength = 'Strong' if current_price < fib_prices[1] else 'Moderate'
    elif current_price > fib_prices[4]:  # 0.618 수준 위
        signal = 'Sell'
        strength = 'Strong' if current_price > fib_prices[5] else 'Moderate'
    else:
        signal = 'Neutral'
        strength = 'Weak'
    
    return {
        'signal': signal,
        'strength': strength,
        'current_level': f"{lower_level:.3f} - {upper_level:.3f}"
    }

def get_overall_signal(bb, ma, rsi, macd, fibonacci):
    """
    모든 분석 결과를 종합하여 전체적인 신호를 결정합니다.
    """
    signals = [bb['signal'], rsi['signal'], macd['signal'], fibonacci['signal']]
    strengths = [bb['strength'], rsi['strength'], macd['strength'], fibonacci['strength']]
    
    buy_count = signals.count('Buy')
    sell_count = signals.count('Sell')
    
    if buy_count > sell_count:
        signal = 'Buy'
    elif sell_count > buy_count:
        signal = 'Sell'
    else:
        signal = 'Neutral'
    
    strong_count = strengths.count('Strong')
    weak_count = strengths.count('Weak')
    
    if strong_count > weak_count:
        strength = 'Strong'
    elif weak_count > strong_count:
        strength = 'Weak'
    else:
        strength = 'Moderate'
    
    return {'signal': signal, 'strength': strength}
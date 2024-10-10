import pandas as pd
import numpy as np

def calculate_bollinger_bands(data, window=20, num_std=2):
    data['MA20'] = data['close'].rolling(window=window).mean()
    data['std_20'] = data['close'].rolling(window=window).std()
    data['upper_band_20'] = data['MA20'] + (data['std_20'] * num_std)
    data['lower_band_20'] = data['MA20'] - (data['std_20'] * num_std)
    return data

def calculate_volume_super_trend_ai(data, period=10, multiplier=3):
    """
    Volume Super Trend AI 지표를 계산합니다.
    
    :param data: OHLCV 데이터가 포함된 DataFrame
    :param period: 기간 (기본값: 10)
    :param multiplier: ATR 승수 (기본값: 3)
    :return: Volume Super Trend AI 값이 추가된 DataFrame
    """
    # 데이터의 복사본 생성
    df = data.copy()
    
    # ATR 계산
    df['TR'] = np.maximum(df['high'] - df['low'], 
                          np.maximum(abs(df['high'] - df['close'].shift(1)),
                                     abs(df['low'] - df['close'].shift(1))))
    df['ATR'] = df['TR'].rolling(window=period).mean()
    
    # Volume Factor 계산
    df['Volume_MA'] = df['volume'].rolling(window=period).mean()
    df['Volume_Factor'] = df['volume'] / df['Volume_MA']
    
    # Super Trend 계산
    df['Upper_Band'] = ((df['high'] + df['low']) / 2) + (multiplier * df['ATR'] * df['Volume_Factor'])
    df['Lower_Band'] = ((df['high'] + df['low']) / 2) - (multiplier * df['ATR'] * df['Volume_Factor'])
    
    df['Super_Trend'] = 0.0
    for i in range(period, len(df)):
        if df['close'].iloc[i] > df['Upper_Band'].iloc[i-1]:
            df.loc[df.index[i], 'Super_Trend'] = df['Lower_Band'].iloc[i]
        elif df['close'].iloc[i] < df['Lower_Band'].iloc[i-1]:
            df.loc[df.index[i], 'Super_Trend'] = df['Upper_Band'].iloc[i]
        else:
            df.loc[df.index[i], 'Super_Trend'] = df['Super_Trend'].iloc[i-1]
    
    # AI 부분: 추세 강도 계산
    df['Trend_Strength'] = abs(df['close'] - df['Super_Trend']) / (df['ATR'] * df['Volume_Factor'])
    
    return df

def analyze_volume_super_trend_ai(data):
    """
    Volume Super Trend AI 분석을 수행합니다.
    """
    latest_close = data['close'].iloc[-1]
    latest_super_trend = data['Super_Trend'].iloc[-1]
    latest_trend_strength = data['Trend_Strength'].iloc[-1]
    
    if latest_close > latest_super_trend:
        signal = 'Buy'
    else:
        signal = 'Sell'
    
    if latest_trend_strength > 1.5:
        strength = 'Strong'
    elif latest_trend_strength > 0.5:
        strength = 'Moderate'
    else:
        strength = 'Weak'
    
    return {
        'signal': signal,
        'strength': strength,
        'super_trend_value': latest_super_trend,
        'trend_strength': latest_trend_strength
    }

def perform_analysis(data):
    """
    주어진 데이터에 대해 기술적 분석을 수행합니다.
    """
    if data.empty:
        print("분석할 데이터가 없습니다.")
        return {}

    # 볼린저 밴드 계산
    data = calculate_bollinger_bands(data)

    # 이동평균선 계산
    data['MA50'] = data['close'].rolling(window=50).mean()
    data['MA200'] = data['close'].rolling(window=200).mean()

    # Volume Super Trend AI 계산
    data = calculate_volume_super_trend_ai(data)
    
    # 각종 분석 수행
    bb_analysis = analyze_bollinger_bands(data)
    ma_analysis = analyze_moving_averages(data)
    rsi_analysis = analyze_rsi(data)
    macd_analysis = analyze_macd(data)
    candlestick_analysis = analyze_candlestick_patterns(data)
    stochastic_analysis = analyze_stochastic(data)
    fibonacci_analysis = analyze_fibonacci_levels(data)
    rapid_fall_rebound_analysis = analyze_rapid_fall_and_rebound(data)
    vsta_analysis = analyze_volume_super_trend_ai(data)
    
    # 전체 분석 결과 종합
    analysis_result = {
        'current_price': data['close'].iloc[-1],
        'bollinger_bands': bb_analysis,
        'moving_averages': ma_analysis,
        'rsi': rsi_analysis,
        'macd': macd_analysis,
        'candlestick_patterns': candlestick_analysis,
        'stochastic': stochastic_analysis,
        'fibonacci': fibonacci_analysis,
        'rapid_fall_rebound': rapid_fall_rebound_analysis,
        'volume_super_trend_ai': vsta_analysis,
        'overall_signal': get_overall_signal(bb_analysis, ma_analysis, rsi_analysis, macd_analysis, 
                                             candlestick_analysis, stochastic_analysis, 
                                             fibonacci_analysis, rapid_fall_rebound_analysis,
                                             vsta_analysis)
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
    ma20 = data['MA20'].iloc[-1]
    ma50 = data['MA50'].iloc[-1]
    ma200 = data['MA200'].iloc[-1]
    current_price = data['close'].iloc[-1]
    
    short_term = 'Bullish' if current_price > ma20 else 'Bearish'
    medium_term = 'Bullish' if ma20 > ma50 else 'Bearish'
    long_term = 'Bullish' if ma50 > ma200 else 'Bearish'
    
    return {
        'short_term': short_term,
        'medium_term': medium_term,
        'long_term': long_term
    }

def analyze_rsi(data, period=14):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
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

def analyze_macd(data, short_period=12, long_period=26, signal_period=9):
    data['EMA_short'] = data['close'].ewm(span=short_period, adjust=False).mean()
    data['EMA_long'] = data['close'].ewm(span=long_period, adjust=False).mean()
    data['MACD'] = data['EMA_short'] - data['EMA_long']
    data['MACD_signal'] = data['MACD'].ewm(span=signal_period, adjust=False).mean()
    
    macd = data['MACD'].iloc[-1]
    signal = data['MACD_signal'].iloc[-1]
    
    if macd > signal:
        return {'signal': 'Buy', 'strength': 'Strong' if macd > 0 else 'Weak'}
    else:
        return {'signal': 'Sell', 'strength': 'Strong' if macd < 0 else 'Weak'}

def analyze_fibonacci_levels(data):
    high = data['high'].max()
    low = data['low'].min()
    current_price = data['close'].iloc[-1]
    
    fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
    fib_prices = [low + (high - low) * level for level in fib_levels]
    
    for i in range(len(fib_prices) - 1):
        if fib_prices[i] <= current_price < fib_prices[i + 1]:
            lower_level = fib_levels[i]
            upper_level = fib_levels[i + 1]
            break
    else:
        return {'signal': 'Neutral', 'strength': 'Weak'}
    
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

def analyze_stochastic(data, k_period=14, d_period=3):
    # 스토캐스틱 %K 계산
    low_min = data['low'].rolling(window=k_period).min()
    high_max = data['high'].rolling(window=k_period).max()
    data['%K'] = (data['close'] - low_min) / (high_max - low_min) * 100

    # 스토캐스틱 %D 계산 (%K의 3일 이동평균)
    data['%D'] = data['%K'].rolling(window=d_period).mean()

    current_k = data['%K'].iloc[-1]
    current_d = data['%D'].iloc[-1]

    if current_k > 80 and current_d > 80:
        signal = 'Sell'
        strength = 'Strong'
    elif current_k < 20 and current_d < 20:
        signal = 'Buy'
        strength = 'Strong'
    elif current_k > current_d and current_k > 50:
        signal = 'Buy'
        strength = 'Moderate'
    elif current_k < current_d and current_k < 50:
        signal = 'Sell'
        strength = 'Moderate'
    else:
        signal = 'Neutral'
        strength = 'Weak'

    return {
        'signal': signal,
        'strength': strength,
        'current_k': current_k,
        'current_d': current_d
    }

def analyze_candlestick_patterns(data):
    recent_candles = data.tail(5)
    bullish_patterns = 0
    bearish_patterns = 0

    if (recent_candles.iloc[-1]['close'] > recent_candles.iloc[-1]['open'] and
        (recent_candles.iloc[-1]['high'] - recent_candles.iloc[-1]['close']) < (recent_candles.iloc[-1]['close'] - recent_candles.iloc[-1]['low']) * 2):
        bullish_patterns += 1
    elif (recent_candles.iloc[-1]['open'] > recent_candles.iloc[-1]['close'] and
          (recent_candles.iloc[-1]['high'] - recent_candles.iloc[-1]['open']) > (recent_candles.iloc[-1]['open'] - recent_candles.iloc[-1]['low']) * 2):
        bearish_patterns += 1

    if recent_candles.iloc[-1]['close'] > recent_candles.iloc[-1]['open'] * 1.02:
        bullish_patterns += 1
    elif recent_candles.iloc[-1]['open'] > recent_candles.iloc[-1]['close'] * 1.02:
        bearish_patterns += 1

    if bullish_patterns > bearish_patterns:
        return {'signal': 'Buy', 'strength': 'Strong' if bullish_patterns > 1 else 'Moderate'}
    elif bearish_patterns > bullish_patterns:
        return {'signal': 'Sell', 'strength': 'Strong' if bearish_patterns > 1 else 'Moderate'}
    else:
        return {'signal': 'Neutral', 'strength': 'Weak'}

def analyze_rapid_fall_and_rebound(data, fall_threshold=-0.5, rebound_threshold=0.2, lookback_period=5):
    recent_data = data.tail(lookback_period + 1).copy()  # copy() 추가
    recent_data['pct_change'] = recent_data['close'].pct_change() * 100
    
    rapid_fall = recent_data['pct_change'].iloc[-2] <= fall_threshold
    rebound = recent_data['pct_change'].iloc[-1] >= rebound_threshold
    
    if rapid_fall and rebound:
        return {
            'signal': 'Buy',
            'strength': 'Strong',
            'reason': f"급락 후 반등 감지: 이전 봉 {recent_data['pct_change'].iloc[-2]:.2f}% 하락, 현재 봉 {recent_data['pct_change'].iloc[-1]:.2f}% 상승"
        }
    elif rapid_fall:
        return {
            'signal': 'Watch',
            'strength': 'Moderate',
            'reason': f"급락 감지: 이전 봉 {recent_data['pct_change'].iloc[-2]:.2f}% 하락, 반등 대기 중"
        }
    else:
        return {
            'signal': 'Neutral',
            'strength': 'Weak',
            'reason': "급락 또는 반등 패턴 없음"
        }
    
def get_overall_signal(bb, ma, rsi, macd, candlestick, stochastic, fibonacci, rapid_fall_rebound, vsta):
    signals = [bb['signal'], rsi['signal'], macd['signal'], candlestick['signal'], 
               stochastic['signal'], fibonacci['signal'], rapid_fall_rebound['signal'],
               vsta['signal']]
    strengths = [bb['strength'], rsi['strength'], macd['strength'], candlestick['strength'], 
                 stochastic['strength'], fibonacci['strength'], rapid_fall_rebound['strength'],
                 vsta['strength']]
    
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
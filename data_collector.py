import requests
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import math
from datetime import datetime, timedelta
from fredapi import Fred

# .env 파일에서 환경 변수 로드
load_dotenv()

# API 키 가져오기
UPBIT_ACCESS_KEY = os.getenv('UPBIT_ACCESS_KEY')
UPBIT_SECRET_KEY = os.getenv('UPBIT_SECRET_KEY')
FRED_API_KEY = os.getenv('FRED_API_KEY')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

# 차트 이미지를 저장할 디렉토리 설정
CHART_DIR = 'chart_images'

# 차트 이미지 디렉토리가 없으면 생성
if not os.path.exists(CHART_DIR):
    os.makedirs(CHART_DIR)

def get_current_price(coin):
    """
    업비트 API를 사용하여 특정 코인의 현재 가격을 가져옵니다.
    """
    url = f"https://api.upbit.com/v1/ticker?markets=KRW-{coin}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data and isinstance(data, list) and len(data) > 0:
            return data[0]['trade_price']
    print(f"Failed to get current price for {coin}. Status code: {response.status_code}")
    return None

def add_technical_indicators(df):
    # 볼린저 밴드 (20시간, 2 표준편차)
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['std_20'] = df['close'].rolling(window=20).std()
    df['upper_band_20'] = df['MA20'] + (df['std_20'] * 2)
    df['lower_band_20'] = df['MA20'] - (df['std_20'] * 2)
    df['bb_position'] = (df['close'] - df['lower_band_20']) / (df['upper_band_20'] - df['lower_band_20'])

    # RSI (14시간)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 이동평균선
    df['MA50'] = df['close'].rolling(window=50).mean()
    df['MA200'] = df['close'].rolling(window=200).mean()

    # MACD 계산 추가
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 피보나치 수준 계산을 위한 트렌드 라인, 지지선, 저항선 설정
    df = calculate_trend_lines_and_levels(df)

    # 피보나치 되돌림 수준 계산 (0.236, 0.382, 0.5, 0.618, 0.786)
    df['fib_high'] = df['trend_high']
    df['fib_low'] = df['trend_low']
    df['fib_0.236'] = df['fib_low'] + (df['fib_high'] - df['fib_low']) * 0.236
    df['fib_0.382'] = df['fib_low'] + (df['fib_high'] - df['fib_low']) * 0.382
    df['fib_0.500'] = df['fib_low'] + (df['fib_high'] - df['fib_low']) * 0.500
    df['fib_0.618'] = df['fib_low'] + (df['fib_high'] - df['fib_low']) * 0.618
    df['fib_0.786'] = df['fib_low'] + (df['fib_high'] - df['fib_low']) * 0.786

    return df

def calculate_trend_lines_and_levels(df):
    """
    데이터 프레임에서 주요 트렌드 라인, 지지선, 저항선을 계산합니다.
    """
    # 최근 n개 기간 동안의 최고가와 최저가 찾기 (n은 가변적이며 필요에 따라 조정 가능)
    n = 50  # 최근 50개 데이터 기준으로 트렌드 라인 계산
    df['trend_high'] = df['high'].rolling(window=n).max()  # 최근 n개 캔들 중 최고가
    df['trend_low'] = df['low'].rolling(window=n).min()  # 최근 n개 캔들 중 최저가

    # 지지선 및 저항선 탐지
    support_levels = []
    resistance_levels = []

    for i in range(2, len(df)):
        if df['low'].iloc[i] < df['low'].iloc[i - 1] and df['low'].iloc[i] < df['low'].iloc[i - 2]:
            support_levels.append(df['low'].iloc[i])
        if df['high'].iloc[i] > df['high'].iloc[i - 1] and df['high'].iloc[i] > df['high'].iloc[i - 2]:
            resistance_levels.append(df['high'].iloc[i])

    # 데이터프레임에 지지 및 저항 수준 추가
    df['support_level'] = np.mean(support_levels[-n:]) if len(support_levels) >= n else np.nan
    df['resistance_level'] = np.mean(resistance_levels[-n:]) if len(resistance_levels) >= n else np.nan

    # 트렌드 라인 설정: 트렌드 방향에 따라 지지선/저항선을 고점 및 저점으로 설정
    df['trend_high'] = df[['trend_high', 'resistance_level']].max(axis=1)
    df['trend_low'] = df[['trend_low', 'support_level']].min(axis=1)

    return df

def collect_upbit_data(coin):
    """
    업비트 API를 사용하여 특정 코인의 1시간봉 데이터를 1달 기간 동안 수집합니다.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # 1달 동안의 1시간봉 개수 계산 (대략 30일 * 24시간 = 720)
    candle_count = math.ceil((end_date - start_date).total_seconds() / 3600)
    
    url = f"https://api.upbit.com/v1/candles/minutes/60?market=KRW-{coin}&count={candle_count}"
    headers = {"Accept": "application/json"}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['candle_date_time_kst'])
        df = df.set_index('timestamp')
        df = df.sort_index()
        
        # 컬럼 이름 변경
        df = df.rename(columns={
            'opening_price': 'open',
            'high_price': 'high',
            'low_price': 'low',
            'trade_price': 'close',
            'candle_acc_trade_volume': 'volume'
        })
        
        # 1달 이전의 데이터 필터링
        df = df[df.index > start_date]
        
        # 기술적 지표 추가
        df = add_technical_indicators(df)
        
        return df
    else:
        print(f"Failed to collect data for {coin}. Status code: {response.status_code}")
        return pd.DataFrame()

def save_chart_image(df, coin):
    """
    주어진 데이터로 차트 이미지를 생성하고 저장합니다.
    """
    filename = f'chart_{coin}_{datetime.now().strftime("%Y%m%d%H%M")}.png'
    filepath = os.path.join(CHART_DIR, filename)
    mpf.plot(df, type='candle', style='charles',
             title=f'{coin} Price Chart',
             ylabel='Price (KRW)',
             ylabel_lower='Volume',
             volume=True,
             savefig=filepath)
    
    print(f"Chart saved: {filepath}")
    
    # 오래된 차트 이미지 삭제
    delete_old_charts(coin)

def delete_old_charts(coin, days_to_keep=7):
    """
    지정된 일수보다 오래된 차트 이미지를 삭제합니다.
    """
    current_time = datetime.now()
    for file in os.listdir(CHART_DIR):
        if file.startswith(f'chart_{coin}_') and file.endswith('.png'):
            filepath = os.path.join(CHART_DIR, file)
            file_date_str = file.split('_')[2].split('.')[0]
            file_date = datetime.strptime(file_date_str, "%Y%m%d%H%M")
            if (current_time - file_date) > timedelta(days=days_to_keep):
                os.remove(filepath)
                print(f"Deleted old chart: {filepath}")

def get_upbit_coin_info(coins=['KRW-BTC', 'KRW-ETH', 'KRW-XRP', 'KRW-SOL', 'KRW-SUI']):
    url = "https://api.upbit.com/v1/ticker"
    
    try:
        params = {"markets": ",".join(coins)}
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        coin_info = []
        for item in data:
            coin_info.append({
                'coin': item['market'],
                'current_price': item['trade_price'],
                'change_rate': item['signed_change_rate'] * 100,
                'high_price': item['high_price'],
                'low_price': item['low_price'],
                'volume': item['acc_trade_volume_24h'],
                'timestamp': datetime.fromtimestamp(item['timestamp'] / 1000)
            })
        
        coin_df = pd.DataFrame(coin_info)
        return coin_df
    
    except Exception as e:
        print(f"업비트 코인 정보 수집 중 오류 발생: {e}")
        return pd.DataFrame()

def get_us_economic_news():
    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "country": "us",
        "category": "business",
        "apiKey": NEWS_API_KEY
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        news_data = []
        for article in data.get("articles", []):
            news_data.append({
                'title': article['title'],
                'description': article['description'],
                'published_at': article['publishedAt']
            })
        
        return news_data
    except Exception as e:
        print(f"미국 경제 뉴스 데이터를 가져오는 데 실패했습니다: {e}")
        return []

def get_fred_economic_data():
    fred = Fred(api_key=FRED_API_KEY)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1년치 데이터로 확장
    
    indicators = {
        'GDP': 'GDP',
        'Inflation': 'CPIAUCSL',
        'Unemployment': 'UNRATE',
        'Interest Rate': 'FEDFUNDS',
        'USD/KRW Exchange Rate': 'DEXKOUS'
    }
    
    economic_data = {}
    
    for name, series_id in indicators.items():
        try:
            data = fred.get_series(series_id, start_date, end_date)
            if not data.empty:
                latest_date = data.index[-1]
                economic_data[name] = {
                    'value': data.iloc[-1],
                    'date': latest_date.strftime('%Y-%m-%d')
                }
            else:
                print(f"{name} 데이터가 비어 있습니다.")
        except Exception as e:
            print(f"{name} 데이터를 가져오는 데 실패했습니다: {e}")
    
    return economic_data

def collect_data():
    print("데이터 수집 중...")
    coin_info = get_upbit_coin_info()
    us_economic_news = get_us_economic_news()
    economic_data = get_fred_economic_data()
    
    if not coin_info.empty:
        print("데이터 수집 완료")
        return {
            "coin_info": coin_info.to_dict('records'),
            "us_economic_news": us_economic_news,
            "economic_data": economic_data
        }
    else:
        print("코인 정보를 가져오지 못했습니다.")
        return {
            "coin_info": None, 
            "us_economic_news": us_economic_news, 
            "economic_data": economic_data
        }

# 테스트를 위한 코드
if __name__ == "__main__":
    collected_data = collect_data()
    if collected_data['coin_info']:
        print("\n수집된 코인 정보:")
        print(pd.DataFrame(collected_data['coin_info']))
    else:
        print("코인 정보 수집 실패")

    print("\n수집된 미국 경제 뉴스 (최근 5개):")
    for news in collected_data['us_economic_news'][:5]:
        print(f"- {news['title']} ({news['published_at']})")

    print("\n수집된 경제 데이터:")
    for indicator, data in collected_data['economic_data'].items():
        if data:
            print(f"- {indicator}: {data['value']} (Date: {data['date']})")
        else:
            print(f"- {indicator}: 데이터 없음")
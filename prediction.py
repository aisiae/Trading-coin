import os
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta
import logging
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
import schedule
import talib  # 새로 추가된 import

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 환경 변수 설정
SENDER_EMAIL = "jjunge1013@gmail.com"
RECEIVER_EMAIL = "jjunge1013@gmail.com"
EMAIL_PASSWORD = "xvnr zytt bpxz pakj"
# TRADING_ECONOMICS_API_KEY = "YOUR_API_KEY"  # TRADING_ECONOMICS API 키 (주석 처리)

# 데이터 저장 경로
DATA_PATH = 'data'
os.makedirs(DATA_PATH, exist_ok=True)

# 예측할 코인 리스트
COINS = ['KRW-BTC', 'KRW-SOL', 'KRW-SUI', 'KRW-ETH']

# JSON 인코딩 함수 (float32 처리를 위한)
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

def send_email_notification(subject, html_content):
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        msg['Subject'] = subject

        msg.attach(MIMEText(html_content, 'html'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, EMAIL_PASSWORD)
        server.send_message(msg)
        logging.info("이메일 알림이 성공적으로 전송되었습니다.")
    except Exception as e:
        logging.error(f"이메일 전송 중 오류 발생: {e}")
    finally:
        server.quit()

def get_current_price(ticker):
    url = f"https://api.upbit.com/v1/ticker?markets={ticker}"
    response = requests.get(url)
    data = response.json()
    return data[0]['trade_price']

def get_historical_data(ticker, count=200):
    url = f"https://api.upbit.com/v1/candles/minutes/60?market={ticker}&count={count}"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data)
    df['datetime'] = pd.to_datetime(df['candle_date_time_kst'])
    df.set_index('datetime', inplace=True)
    return df[['opening_price', 'high_price', 'low_price', 'trade_price', 'candle_acc_trade_volume']].sort_index()

# TRADING_ECONOMICS 관련 함수 (주석 처리)
"""
def get_economic_data():
    url = f"https://api.tradingeconomics.com/news?c={TRADING_ECONOMICS_API_KEY}"
    response = requests.get(url)
    return response.json()

def get_economic_calendar():
    url = f"https://api.tradingeconomics.com/calendar?c={TRADING_ECONOMICS_API_KEY}"
    response = requests.get(url)
    return response.json()
"""

def prepare_data(data, scaler):
    scaled_data = scaler.fit_transform(data[['trade_price', 'candle_acc_trade_volume']])
    x, y = [], []
    for i in range(60, len(scaled_data)):
        x.append(scaled_data[i-60:i])
        y.append(scaled_data[i, 0])
    return np.array(x), np.array(y)

def create_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def calculate_prediction_accuracy(predicted, actual):
    if predicted is None or actual is None or predicted == 0:
        return None
    return (actual - predicted) / predicted * 100

# 새로 추가된 함수: 기술적 지표 계산
def calculate_technical_indicators(df):
    close_prices = df['trade_price'].values
    high_prices = df['high_price'].values
    low_prices = df['low_price'].values
    volume = df['candle_acc_trade_volume'].values

    # 이동평균선
    df['SMA_10'] = talib.SMA(close_prices, timeperiod=10)
    df['SMA_20'] = talib.SMA(close_prices, timeperiod=20)
    df['SMA_50'] = talib.SMA(close_prices, timeperiod=50)

    # MACD
    df['MACD'], df['MACD_Signal'], _ = talib.MACD(close_prices)

    # RSI
    df['RSI'] = talib.RSI(close_prices)

    # 볼린저 밴드
    df['Upper_Band'], df['Middle_Band'], df['Lower_Band'] = talib.BBANDS(close_prices)

    # 스토캐스틱 오실레이터
    df['SlowK'], df['SlowD'] = talib.STOCH(high_prices, low_prices, close_prices)

    return df

# 새로 추가된 함수: 기술적 지표 해석
def interpret_indicators(df):
    last_row = df.iloc[-1]
    current_price = last_row['trade_price']

    interpretation = {}

    # 이동평균선 해석
    if current_price > last_row['SMA_50'] > last_row['SMA_20'] > last_row['SMA_10']:
        interpretation['MA_Trend'] = "강한 상승세"
    elif current_price > last_row['SMA_10'] > last_row['SMA_20'] > last_row['SMA_50']:
        interpretation['MA_Trend'] = "상승세"
    elif current_price < last_row['SMA_50'] < last_row['SMA_20'] < last_row['SMA_10']:
        interpretation['MA_Trend'] = "강한 하락세"
    elif current_price < last_row['SMA_10'] < last_row['SMA_20'] < last_row['SMA_50']:
        interpretation['MA_Trend'] = "하락세"
    else:
        interpretation['MA_Trend'] = "중립"

    # MACD 해석
    if last_row['MACD'] > last_row['MACD_Signal']:
        interpretation['MACD'] = "매수 신호"
    else:
        interpretation['MACD'] = "매도 신호"

    # RSI 해석
    if last_row['RSI'] > 70:
        interpretation['RSI'] = "과매수"
    elif last_row['RSI'] < 30:
        interpretation['RSI'] = "과매도"
    else:
        interpretation['RSI'] = "중립"

    # 볼린저 밴드 해석
    if current_price > last_row['Upper_Band']:
        interpretation['Bollinger'] = "과매수"
    elif current_price < last_row['Lower_Band']:
        interpretation['Bollinger'] = "과매도"
    else:
        interpretation['Bollinger'] = "중립"

    # 스토캐스틱 오실레이터 해석
    if last_row['SlowK'] > 80 and last_row['SlowD'] > 80:
        interpretation['Stochastic'] = "과매수"
    elif last_row['SlowK'] < 20 and last_row['SlowD'] < 20:
        interpretation['Stochastic'] = "과매도"
    else:
        interpretation['Stochastic'] = "중립"

    return interpretation

def predict_price(coin):
    try:
        # 데이터 가져오기
        historical_data = get_historical_data(coin)
        current_price = get_current_price(coin)

        # 기술적 지표 계산 및 해석 (새로 추가됨)
        historical_data = calculate_technical_indicators(historical_data)
        tech_interpretation = interpret_indicators(historical_data)

        # 이전 예측 데이터 로드
        previous_predictions_file = os.path.join(DATA_PATH, f'{coin}_previous_predictions.json')
        if os.path.exists(previous_predictions_file):
            with open(previous_predictions_file, 'r') as f:
                previous_predictions = json.load(f)
        else:
            previous_predictions = {}

        # 정확도 계산
        accuracy_1h = calculate_prediction_accuracy(previous_predictions.get("1h"), current_price)
        accuracy_24h = calculate_prediction_accuracy(previous_predictions.get("24h"), current_price)

        # 데이터 전처리
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(historical_data[['trade_price', 'candle_acc_trade_volume']])
        
        x, y = prepare_data(historical_data, scaler)

        # 모델 생성 및 학습
        model = create_model((x.shape[1], x.shape[2]))
        model.fit(x, y, epochs=50, batch_size=32, verbose=0)

        # 1시간 후와 24시간 후 예측
        last_60_hours = scaled_data[-60:]
        last_60_hours_reshaped = np.reshape(last_60_hours, (1, last_60_hours.shape[0], last_60_hours.shape[1]))
        
        # 1시간 후 예측
        prediction_1h = model.predict(last_60_hours_reshaped)
        predicted_price_1h = scaler.inverse_transform(np.column_stack((prediction_1h, np.zeros_like(prediction_1h))))[0][0]
        
        # 24시간 후 예측
        prediction_24h = last_60_hours_reshaped.copy()
        for _ in range(24):
            next_pred = model.predict(prediction_24h[:, -60:, :])
            next_pred_reshaped = np.reshape(next_pred, (1, 1, 1))
            zeros_column = np.zeros((1, 1, 1))
            next_hour = np.concatenate([next_pred_reshaped, zeros_column], axis=2)
            prediction_24h = np.concatenate([prediction_24h, next_hour], axis=1)
        
        predicted_price_24h = scaler.inverse_transform(prediction_24h[0, -1:, :])[:, 0][0]

        # 기술적 지표를 기반으로 예측 조정 (새로 추가됨)
        if tech_interpretation['MA_Trend'] in ["강한 상승세", "상승세"] and tech_interpretation['MACD'] == "매수 신호":
            predicted_price_1h *= 1.01  # 1% 상향 조정
            predicted_price_24h *= 1.03  # 3% 상향 조정
        elif tech_interpretation['MA_Trend'] in ["강한 하락세", "하락세"] and tech_interpretation['MACD'] == "매도 신호":
            predicted_price_1h *= 0.99  # 1% 하향 조정
            predicted_price_24h *= 0.97  # 3% 하향 조정

        # 결과 저장
        result = {
            'timestamp': datetime.now().isoformat(),
            'coin': coin,
            'current_price': float(current_price),
            'predicted_price_1h': float(predicted_price_1h),
            'predicted_price_24h': float(predicted_price_24h),
            'price_trend_1h': "상승 추세" if predicted_price_1h > current_price else "하락 추세",
            'price_trend_24h': "상승 추세" if predicted_price_24h > current_price else "하락 추세",
            'accuracy_1h': accuracy_1h,
            'accuracy_24h': accuracy_24h,
            'previous_prediction_1h': previous_predictions.get("1h"),
            'previous_prediction_24h': previous_predictions.get("24h"),
            'technical_analysis': tech_interpretation  # 새로 추가된 기술적 분석 결과
        }

        # 현재 예측 저장 (다음 실행 시 정확도 계산에 사용)
        with open(previous_predictions_file, 'w') as f:
            json.dump({
                "1h": float(predicted_price_1h),
                "24h": float(predicted_price_24h)
            }, f)

        return result

    except Exception as e:
        logging.error(f"{coin} 예측 중 오류 발생: {str(e)}")
        return None

def run_predictions():
    all_results = []
    for coin in COINS:
        result = predict_price(coin)
        if result:
            all_results.append(result)

    # 모든 결과 저장
    with open(os.path.join(DATA_PATH, 'latest_predictions.json'), 'w') as f:
        json.dump(all_results, f, cls=NumpyEncoder)

    # 히스토리 업데이트
    history_file = os.path.join(DATA_PATH, 'prediction_history.json')
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
    else:
        history = []

    history.append(all_results)
    history = history[-10:]  # 최근 10개만 유지

    with open(history_file, 'w') as f:
        json.dump(history, f, cls=NumpyEncoder)

    # 로깅 및 이메일 알림
    email_content = "<h2>암호화폐 가격 예측 결과:</h2>"
    for result in all_results:
        coin = result['coin']
        current_price = int(result['current_price'])
        predicted_price_1h = int(result['predicted_price_1h'])
        predicted_price_24h = int(result['predicted_price_24h'])
        accuracy_1h = result['accuracy_1h']
        accuracy_24h = result['accuracy_24h']
        tech_analysis = result['technical_analysis']

        logging.info(f"{coin} 예측 완료: 현재 가격 {current_price}, 1시간 후 예측 가격 {predicted_price_1h}, 24시간 후 예측 가격 {predicted_price_24h}")
        
        email_content += f"<h3>코인: {coin}</h3>"
        email_content += f"<p>현재 가격: {current_price:,}</p>"
        email_content += f"<p>1시간 후 예측 가격: <span style='color: {'blue' if predicted_price_1h > current_price else 'red'};'>{predicted_price_1h:,}</span></p>"
        email_content += f"<p>24시간 후 예측 가격: <span style='color: {'blue' if predicted_price_24h > current_price else 'red'};'>{predicted_price_24h:,}</span></p>"
        email_content += f"<p>1시간 추세: {result['price_trend_1h']}</p>"
        email_content += f"<p>24시간 추세: {result['price_trend_24h']}</p>"
        if accuracy_1h is not None:
            email_content += f"<p>1시간 예측 정확도: {accuracy_1h:.2f}%</p>"
        if accuracy_24h is not None:
            email_content += f"<p>24시간 예측 정확도: {accuracy_24h:.2f}%</p>"
        email_content += "<h4>기술적 분석 결과:</h4>"
        email_content += f"<p>이동평균선 추세: {tech_analysis['MA_Trend']}</p>"
        email_content += f"<p>MACD: {tech_analysis['MACD']}</p>"
        email_content += f"<p>RSI: {tech_analysis['RSI']}</p>"
        email_content += f"<p>볼린저 밴드: {tech_analysis['Bollinger']}</p>"
        email_content += f"<p>스토캐스틱: {tech_analysis['Stochastic']}</p>"
        email_content += "<hr>"

    # 이메일 발송
    send_email_notification("암호화폐 가격 예측 결과", email_content)

def main():
    logging.info("암호화폐 가격 예측 프로그램이 시작되었습니다.")
    
    # 즉시 한 번 실행
    run_predictions()
    
    # 1시간마다 실행 예약
    schedule.every(1).hour.do(run_predictions)
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # 1분마다 체크

if __name__ == "__main__":
    main()
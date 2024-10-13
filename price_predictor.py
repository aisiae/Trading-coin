import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
from datetime import datetime, timedelta


if not os.path.exists('predictions'):
    os.makedirs('predictions')


# 예측 결과를 저장할 DataFrame
predictions_df = pd.DataFrame(columns=['timestamp', 'coin', 'current_price', 'predicted_30min', 'predicted_1hour', 'predicted_24hours'])

def save_prediction(coin, current_price, predicted_30min, predicted_1hour, predicted_24hours):
    predictions_file = os.path.join('predictions', f'predictions_{coin}.csv')
    new_prediction = pd.DataFrame({
        'timestamp': [datetime.now()],
        'coin': [coin],
        'current_price': [current_price],
        'predicted_30min': [predicted_30min],
        'predicted_1hour': [predicted_1hour],
        'predicted_24hours': [predicted_24hours]
    })
    
    if os.path.exists(predictions_file):
        predictions_df = pd.read_csv(predictions_file, parse_dates=['timestamp'])
        predictions_df = pd.concat([predictions_df, new_prediction], ignore_index=True)
    else:
        predictions_df = new_prediction
    
    # 7일 이상 된 데이터 삭제
    predictions_df = predictions_df[predictions_df['timestamp'] > datetime.now() - timedelta(days=7)]
    
    predictions_df.to_csv(predictions_file, index=False)

def calculate_error(coin):
    predictions_file = os.path.join('predictions', f'predictions_{coin}.csv')
    if not os.path.exists(predictions_file):
        return None
    
    predictions_df = pd.read_csv(predictions_file, parse_dates=['timestamp'])

    if len(predictions_df) < 2:
        return None

    latest_prediction = predictions_df.iloc[-1]
    previous_prediction = predictions_df.iloc[-2]

    time_diff = latest_prediction['timestamp'] - previous_prediction['timestamp']

    if time_diff < timedelta(minutes=30):
        actual = latest_prediction['current_price']
        predicted = previous_prediction['predicted_30min']
    elif time_diff < timedelta(hours=1):
        actual = latest_prediction['current_price']
        predicted = previous_prediction['predicted_1hour']
    elif time_diff < timedelta(hours=24):
        actual = latest_prediction['current_price']
        predicted = previous_prediction['predicted_24hours']
    else:
        return None

    error = actual - predicted
    error_percentage = (error / predicted) * 100

    return {
        'error': error,
        'error_percentage': error_percentage
    }

def train_model(coin, data):
    """
    주어진 데이터로 랜덤 포레스트 모델을 학습시킵니다.
    """
    # 특성과 타겟 분리
    features = ['bb_position', 'MA20', 'MA50', 'MACD', 'RSI']
    X = data[features]
    y = data['close'].shift(-1)  # 다음 날의 종가를 예측

    # NaN 제거
    X = X.dropna()
    y = y.loc[X.index]  # X와 y의 인덱스를 일치시킵니다

    # y에서 NaN 제거
    mask = ~np.isnan(y)
    X = X.loc[mask]
    y = y.loc[mask]

    # 데이터가 충분한지 확인
    if len(X) < 10:  # 예: 최소 10개의 샘플이 필요하다고 가정
        print("학습에 필요한 충분한 데이터가 없습니다.")
        return None

    # 학습 데이터와 테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 랜덤 포레스트 모델 학습
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 모델 성능 평가
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"모델 MSE: {mse}")

    # 모델 저장
    model_path = os.path.join('predictions', f'price_prediction_model_{coin}.joblib')
    joblib.dump(model, model_path)
    print(f"모델이 {model_path}에 저장되었습니다.")

    return model

def predict_prices(coin, coin_data, analysis_results, current_price, news_data, economic_data):
    """
    학습된 모델을 사용하여 향후 가격을 예측합니다.
    """
    # 모델 로드 (없으면 새로 학습)
    model_path = os.path.join('predictions', f'price_prediction_model_{coin}.joblib')
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"저장된 모델이 없습니다. 새로운 모델을 학습합니다. ({coin})")
        model = train_model(coin, coin_data)
        if model is None:
            print(f"모델 학습에 실패했습니다. ({coin})")
            return None
        
    # 새로운 특성 추가
    latest_data = pd.DataFrame({
        'bb_position': [coin_data['bb_position'].iloc[-1]],
        'MA20': [coin_data['MA20'].iloc[-1]],
        'MA50': [coin_data['MA50'].iloc[-1]],
        'MACD': [coin_data['MACD'].iloc[-1]],
        'RSI': [coin_data['RSI'].iloc[-1]],
        'stoch_k': [coin_data['%K'].iloc[-1] if '%K' in coin_data.columns else 0],
        'stoch_d': [coin_data['%D'].iloc[-1] if '%D' in coin_data.columns else 0],
        'vsta_trend_strength': [analysis_results['volume_super_trend_ai']['trend_strength']],
        'rapid_fall_rebound': [1 if analysis_results['rapid_fall_rebound']['signal'] == 'Buy' else 0]
    })

    # 예측을 위한 데이터 준비
    latest_data = pd.DataFrame({
        'bb_position': [coin_data['bb_position'].iloc[-1]],
        'MA20': [coin_data['MA20'].iloc[-1]],
        'MA50': [coin_data['MA50'].iloc[-1]],
        'MACD': [coin_data['MACD'].iloc[-1]],
        'RSI': [coin_data['RSI'].iloc[-1]]
    })

    # NaN 값 확인 및 처리
    if latest_data.isnull().values.any():
        print("예측 데이터에 NaN 값이 포함되어 있습니다. 이를 0으로 대체합니다.")
        latest_data = latest_data.fillna(0)

    # 가격 예측 (과도한 변동 방지)
    predicted_price = model.predict(latest_data)[0]
    max_change = 0.05  # 최대 5% 변동 허용

    predicted_price_30min = max(min(predicted_price, current_price * (1 + max_change)), current_price * (1 - max_change))
    predicted_price_1hour = max(min(predicted_price * 1.001, current_price * (1 + max_change)), current_price * (1 - max_change))
    predicted_price_24hours = max(min(predicted_price * 1.005, current_price * (1 + max_change)), current_price * (1 - max_change))

    # 예측 결과 저장
    save_prediction(coin, current_price, predicted_price_30min, predicted_price_1hour, predicted_price_24hours)

    # 오차 계산
    error_info = calculate_error(coin)
    if error_info:
        prediction_results = {
            'current_price': current_price,
            'predicted_price_30min': round(predicted_price_30min, 2),
            'predicted_price_1hour': round(predicted_price_1hour, 2),
            'predicted_price_24hours': round(predicted_price_24hours, 2),
            'last_error': error_info['error'],
            'last_error_percentage': error_info['error_percentage']
        }
    else:
        prediction_results = {
            'current_price': current_price,
            'predicted_price_30min': round(predicted_price_30min, 2),
            'predicted_price_1hour': round(predicted_price_1hour, 2),
            'predicted_price_24hours': round(predicted_price_24hours, 2)
        }

    # 예측 이유
    if predicted_price > current_price:
        prediction_results['reason'] = "기술적 지표가 상승세를 나타내고 있습니다."
    else:
        prediction_results['reason'] = "기술적 지표가 하락세를 나타내고 있습니다."

     # 예측 이유 업데이트
    prediction_results['reason'] += f" Volume Super Trend AI는 {analysis_results['volume_super_trend_ai']['signal']} 신호를 보이고 있으며, 강도는 {analysis_results['volume_super_trend_ai']['strength']}입니다."
    prediction_results['reason'] += f" 급락 후 반등 분석은 {analysis_results['rapid_fall_rebound']['signal']} 신호를 보이고 있습니다."

    print("가격 예측 완료")
    return prediction_results

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        coin = sys.argv[1]
    else:
        coin = "BTC"  # 기본값으로 BTC 사용

    predictions_file = os.path.join('predictions', f'predictions_{coin}.csv')
    if os.path.exists(predictions_file):
        predictions_df = pd.read_csv(predictions_file, parse_dates=['timestamp'])
        print(f"\n최근 {coin} 예측 데이터:")
        print(predictions_df.tail())
    else:
        print(f"{coin}에 대한 예측 데이터가 없습니다.")
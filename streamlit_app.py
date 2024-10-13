import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data_collector import collect_data, collect_upbit_data
from technical_analysis import perform_analysis
from price_predictor import predict_prices
from datetime import datetime
import json
import os

# 페이지 설정
st.set_page_config(layout="wide", page_title="코인 가격 예측 대시보드")

# CSS를 사용하여 전체 화면 스타일 적용
st.markdown("""
<style>
    .reportview-container {
        margin-top: -2em;
    }
    .reportview-container .main .block-container {
        max-width: 1000px;
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# 최신 예측 데이터 로드
def load_latest_predictions():
    try:
        # 실제 파일 경로가 정확한지 확인
        prediction_files = [f for f in os.listdir('predictions') if f.startswith("predictions_") and f.endswith(".json")]
        if not prediction_files:
            st.warning("예측 데이터 파일이 없습니다.")
            return None
        latest_file = max(prediction_files)
        with open(f'predictions/{latest_file}', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("예측 데이터를 찾을 수 없습니다. predictions 폴더를 확인하세요.")
        return None
    except json.JSONDecodeError:
        st.error("예측 데이터를 읽는 중 오류가 발생했습니다. 파일 형식을 확인하세요.")
        return None


# 차트 생성 함수
def create_technical_chart(df):
    df = df.tail(72)  # 1시간 봉 * 24시간 * 3일 = 72
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='캔들스틱'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['upper_band_20'], name='상단 밴드 (20, 2)', line=dict(color='rgba(255, 0, 0, 0.3)')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='20시간 이동평균', line=dict(color='rgba(255, 0, 0, 0.8)')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['lower_band_20'], name='하단 밴드 (20, 2)', line=dict(color='rgba(255, 0, 0, 0.3)')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='50시간 이동평균', line=dict(color='rgba(0, 255, 0, 0.8)')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA200'], name='200시간 이동평균', line=dict(color='rgba(128, 0, 128, 0.8)')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='rgba(255, 165, 0, 0.8)')), row=2, col=1)

    fig.update_layout(title='1시간봉 코인 가격 차트 및 기술적 지표 (최근 3일)', xaxis_title='날짜', yaxis_title='가격', height=800, width=1000)
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_yaxes(title_text='RSI', row=2, col=1)

    return fig

# 피보나치 분석
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
    
    if current_price < fib_prices[2]:
        signal = 'Buy'
        strength = 'Strong' if current_price < fib_prices[1] else 'Moderate'
    elif current_price > fib_prices[4]:
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

# 오차율 스타일 적용 함수
def style_error_rate(val):
    color = 'red' if abs(val) >= 10 else 'white'
    return f'color: {color}'

# 누적 예측 결과 분석 테이블 표시
def display_cumulative_predictions_table(cumulative_predictions):
    st.subheader('누적 예측 결과 분석')
    
    data = []
    for pred_set in cumulative_predictions:
        for coin, predictions in pred_set.items():
            for pred in predictions:
                data.append({
                '코인': coin,  # 코인 이름 추가
                '예측 시간': pred['prediction_time'],
                '30분 예측금액': int(pred['predicted_price_30min']),
                '1시간 예측금액': int(pred['predicted_price_1hour']),
                '24시간 예측금액': int(pred['predicted_price_24hours']),
                '현재 가격': int(pred['current_price']),
                '30분 오차율': round(pred['error_percentage_30min'], 1),
                '1시간 오차율': round(pred['error_percentage_1hour'], 1),
                '24시간 오차율': round(pred['error_percentage_24hours'], 1)
            })

    df = pd.DataFrame(data)
    styled_df = df.style.applymap(style_error_rate, subset=['30분 오차율', '1시간 오차율', '24시간 오차율'])

    st.dataframe(styled_df)

# 누적 예측 데이터를 불러오는 함수
def load_cumulative_predictions():
    all_predictions = []
    prediction_files = [f for f in os.listdir('predictions') if f.startswith("predictions_") and f.endswith(".json")]
    
    for file in prediction_files:
        with open(os.path.join('predictions', file), 'r') as f:
            try:
                data = json.load(f)
                all_predictions.append(data)
            except json.JSONDecodeError:
                st.error(f"Error decoding JSON from {file}")
    
    return all_predictions

# 이전 예측 결과 분석 (30분, 1시간, 24시간 전 예측 결과)
def display_previous_predictions(cumulative_preds, current_price):
    st.subheader('이전 예측 결과 분석')

    if cumulative_preds and len(cumulative_preds) > 1:
        previous_prediction = cumulative_preds[-2]  # 직전 예측 데이터를 불러옴
        prev_time = previous_prediction['prediction_time']

        # 직전 예측 시간 및 현재 가격 표시
        st.write(f"예측 시간: {prev_time}")
        st.write(f"현재 가격: {current_price:,}원")

        # 30분, 1시간, 24시간 후 예측 결과에 대한 분석
        col1, col2, col3 = st.columns(3)
        
        for col, timeframe in zip([col1, col2, col3], ['30min', '1hour', '24hours']):
            prev_predicted_price = int(previous_prediction[f'predicted_price_{timeframe}'])
            error = current_price - prev_predicted_price
            error_percentage = (error / prev_predicted_price) * 100

            col.write(f"<div style='font-size: 24px;'>{prev_predicted_price:,}원</div>", unsafe_allow_html=True)
            col.write(f"<div style='color:red;'>{error:,}원 ({error_percentage:.1f}%)</div>", unsafe_allow_html=True)
    else:
        st.write("이전 예측 데이터가 없습니다.")


def main():
    st.title('코인 가격 예측 대시보드')

    # 최신 예측 데이터를 로드
    predictions = load_latest_predictions()

    if predictions is None:  # 데이터를 불러오지 못한 경우
        st.warning("예측 데이터를 불러오지 못했습니다.")
        return

    # 코인 목록 설정 (여러 코인을 추가)
    coin_options = ['BTC', 'ETH', 'XRP', 'SOL', 'SUI']  
    selected_coin = st.sidebar.radio('코인 선택', coin_options, key="main_coin_selector")
    
    if selected_coin not in predictions:
        st.warning("선택된 코인의 예측 데이터가 없습니다.")
        return

    coin_prediction = predictions[selected_coin]
    current_price = coin_prediction[0]['current_price']

    # 현재 예측 결과 표시
    st.subheader('현재 가격 예측')
    st.write(f"현재 가격: {current_price:,}원")
    st.write(f"예측 시간: {coin_prediction[0]['prediction_time']}")

    col1, col2, col3 = st.columns(3)
    predicted_30min = coin_prediction[0]['predicted_price_30min']
    predicted_1hour = coin_prediction[0]['predicted_price_1hour']
    predicted_24hours = coin_prediction[0]['predicted_price_24hours']

    col1.metric("30분 후 예상 가격", f"{int(predicted_30min):,}원", f"{int(predicted_30min - current_price):,}원")
    col2.metric("1시간 후 예상 가격", f"{int(predicted_1hour):,}원", f"{int(predicted_1hour - current_price):,}원")
    col3.metric("24시간 후 예상 가격", f"{int(predicted_24hours):,}원", f"{int(predicted_24hours - current_price):,}원")

    st.write(f"예측 이유: {coin_prediction[0]['reason']}")

    # 이전 예측 결과 표시
    all_cumulative_preds = load_cumulative_predictions()
    if selected_coin in all_cumulative_preds:
        cumulative_preds = all_cumulative_preds[selected_coin]
        display_previous_predictions(cumulative_preds, current_price)
    else:
        st.warning(f"{selected_coin}에 대한 누적 예측 데이터가 없습니다.")


    # 데이터를 수집하여 차트에 사용 (coin_data와 collected_data를 가져옴)
    collected_data = collect_data()
    coin_data = collect_upbit_data(selected_coin)

    if not coin_data.empty:
        # 차트 표시
        st.subheader('가격 차트 및 기술적 지표')
        chart = create_technical_chart(coin_data)
        st.plotly_chart(chart, use_container_width=True)
    else:
        st.warning('차트를 생성할 데이터가 없습니다.')

    # 경제 지표 데이터 표시
    if 'economic_data' in collected_data and collected_data['economic_data']:
        st.subheader('주요 경제 지표')
        for indicator, data in collected_data['economic_data'].items():
            st.write(f"{indicator}: {data['value']} (날짜: {data['date']})")

    # 뉴스 데이터 표시
    if 'us_economic_news' in collected_data and collected_data['us_economic_news']:
        st.subheader('최근 경제 뉴스')
        for news in collected_data['us_economic_news'][:5]:  # 최근 5개 뉴스만 표시
            st.write(f"- {news['title']} ({news['published_at']})")

    # 누적 예측 데이터를 불러옴
    cumulative_predictions = load_cumulative_predictions()

    # 누적 예측 데이터를 표시
    if cumulative_predictions:
        display_cumulative_predictions_table(cumulative_predictions)
    else:
        st.warning("누적 예측 데이터가 없습니다.")

    # 누적 예측 결과 분석을 제일 아래로 이동
    if selected_coin in all_cumulative_preds:
        display_cumulative_predictions_table({selected_coin: all_cumulative_preds[selected_coin]})
    else:
        st.warning(f"{selected_coin}에 대한 누적 예측 데이터가 없습니다.")

if __name__ == '__main__':
    main()

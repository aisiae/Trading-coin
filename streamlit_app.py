import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data_collector import collect_data, collect_upbit_data
from technical_analysis import perform_analysis
from price_predictor import predict_prices
from datetime import datetime, timedelta
import streamlit as st
import json
import os
from datetime import datetime

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

def load_latest_predictions():
    prediction_files = [f for f in os.listdir() if f.startswith("predictions_") and f.endswith(".json")]
    if not prediction_files:
        return None
    latest_file = max(prediction_files)
    with open(latest_file, 'r') as f:
        return json.load(f)
    
def create_technical_chart(df):
    # 3일치 데이터만 사용
    df = df.tail(72)  # 1시간 봉 * 24시간 * 3일 = 72
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, row_heights=[0.7, 0.3])

    # 캔들스틱 차트
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'],
                                 low=df['low'], close=df['close'], name='캔들스틱'), row=1, col=1)

    # 볼린저 밴드 (20시간, 2 표준편차)
    fig.add_trace(go.Scatter(x=df.index, y=df['upper_band_20'], name='상단 밴드 (20, 2)',
                             line=dict(color='rgba(255, 0, 0, 0.3)')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='20시간 이동평균',
                             line=dict(color='rgba(255, 0, 0, 0.8)')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['lower_band_20'], name='하단 밴드 (20, 2)',
                             line=dict(color='rgba(255, 0, 0, 0.3)')), row=1, col=1)

    # 이동평균선
    fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='50시간 이동평균',
                             line=dict(color='rgba(0, 255, 0, 0.8)')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA200'], name='200시간 이동평균',
                             line=dict(color='rgba(128, 0, 128, 0.8)')), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                             line=dict(color='rgba(255, 165, 0, 0.8)')), row=2, col=1)

    # 레이아웃 설정
    fig.update_layout(title='1시간봉 코인 가격 차트 및 기술적 지표 (최근 3일)', xaxis_title='날짜',
                      yaxis_title='가격', height=800, width=1000)
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_yaxes(title_text='RSI', row=2, col=1)

    return fig

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

def main():
    st.title('코인 가격 예측 대시보드')

    predictions = load_latest_predictions()
    if not predictions:
        st.warning("예측 데이터가 없습니다.")
        return

    coin_options = list(predictions.keys())
    selected_coin = st.sidebar.radio('코인 선택', coin_options)

    st.write(f'선택된 코인: {selected_coin}')

    coin_prediction = predictions[selected_coin]
    
    # 현재 예측 결과 표시
    st.subheader('현재 가격 예측')
    current_price = int(coin_prediction['current_price'])
    st.write(f"현재 가격: {current_price:,}원")
    st.write(f"예측 시간: {coin_prediction['prediction_time']}")
    
    col1, col2, col3 = st.columns(3)
    predicted_30min = int(coin_prediction['predicted_price_30min'])
    predicted_1hour = int(coin_prediction['predicted_price_1hour'])
    predicted_24hours = int(coin_prediction['predicted_price_24hours'])
    
    col1.metric("30분 후 예상 가격", f"{predicted_30min:,}원", f"{predicted_30min - current_price:,}원")
    col2.metric("1시간 후 예상 가격", f"{predicted_1hour:,}원", f"{predicted_1hour - current_price:,}원")
    col3.metric("24시간 후 예상 가격", f"{predicted_24hours:,}원", f"{predicted_24hours - current_price:,}원")

    st.write(f"예측 이유: {coin_prediction['reason']}")

    # 이전 예측 분석 (30분, 1시간, 24시간 전 예측 결과)
    st.subheader('이전 예측 결과 분석')
    if 'previous_predictions' in coin_prediction:
        prev_preds = coin_prediction['previous_predictions']
        
        for timeframe in ['30min', '1hour', '24hours']:
            if timeframe in prev_preds:
                st.write(f"\n{timeframe} 전 예측 결과:")
                prev_pred = prev_preds[timeframe]
                prev_price = int(prev_pred['current_price'])
                prev_predicted_price = int(prev_pred['predicted_price'])
                error = current_price - prev_predicted_price
                error_percentage = (error / prev_predicted_price) * 100
                
                st.write(f"예측 시간: {prev_pred['prediction_time']}")
                st.write(f"{timeframe} 전 현재 가격: {prev_price:,}원")
                st.write(f"{timeframe} 전 예측 가격: {prev_predicted_price:,}원")
                st.write(f"실제 가격과의 오차: {error:,}원 ({error_percentage:.2f}%)")
    else:
        st.write("이전 예측 데이터가 없습니다.")

    # 오차 추이 분석
    st.subheader('오차 추이 분석')
    if 'error_history' in coin_prediction:
        error_df = pd.DataFrame(coin_prediction['error_history'])
        st.line_chart(error_df.set_index('timestamp')['error_percentage'])
    else:
        st.write("오차 추이 데이터가 없습니다.")

    # 데이터 수집
    collected_data = collect_data()
    coin_data = collect_upbit_data(selected_coin)

    if coin_data.empty:
        st.warning('데이터를 불러오는데 실패했습니다.')
        return

    # 코인 정보 표시
    if collected_data['coin_info']:
        coin_info = pd.DataFrame(collected_data['coin_info'])
        selected_coin_info = coin_info[coin_info['coin'] == f'KRW-{selected_coin}'].iloc[0]
        current_price = selected_coin_info['current_price']
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 기술적 분석 및 가격 예측
        analysis_results = perform_analysis(coin_data)
        predictions = predict_prices(selected_coin, coin_data, analysis_results, current_price, collected_data.get('us_economic_news', []), collected_data.get('economic_data', {}))
     
        # 피보나치 분석 결과 추가
        fibonacci_analysis = analyze_fibonacci_levels(coin_data)
        analysis_results['fibonacci'] = fibonacci_analysis

        # 예측 이유 상세히 표시 (모두 한글로 변경)
        st.write("예측 이유:")

        # 볼린저 밴드 위치
        bb_signal = "매수" if analysis_results['bollinger_bands']['signal'] == 'Buy' else "매도" if analysis_results['bollinger_bands']['signal'] == 'Sell' else "중립"
        bb_strength = "강함" if analysis_results['bollinger_bands']['strength'] == 'Strong' else "약함" if analysis_results['bollinger_bands']['strength'] == 'Weak' else "중립"
        st.write(f"- 볼린저 밴드 위치: {bb_signal} ({bb_strength})")

        # 이동 평균선 분석
        ma_short = "상승" if analysis_results['moving_averages']['short_term'] == 'Bullish' else "하락"
        ma_medium = "상승" if analysis_results['moving_averages']['medium_term'] == 'Bullish' else "하락"
        ma_long = "상승" if analysis_results['moving_averages']['long_term'] == 'Bullish' else "하락"
        st.write(f"- 이동 평균선 분석: 단기 추세 ({ma_short}), 중기 추세 ({ma_medium}), 장기 추세 ({ma_long})")

        # RSI 분석
        rsi_signal = "매수" if analysis_results['rsi']['signal'] == 'Buy' else "매도" if analysis_results['rsi']['signal'] == 'Sell' else "중립"
        rsi_strength = "강함" if analysis_results['rsi']['strength'] == 'Strong' else "약함" if analysis_results['rsi']['strength'] == 'Weak' else "중립"
        st.write(f"- RSI 분석: {rsi_signal} ({rsi_strength})")

        # MACD 분석
        macd_signal = "매수" if analysis_results['macd']['signal'] == 'Buy' else "매도" if analysis_results['macd']['signal'] == 'Sell' else "중립"
        macd_strength = "강함" if analysis_results['macd']['strength'] == 'Strong' else "약함" if analysis_results['macd']['strength'] == 'Weak' else "중립"
        st.write(f"- MACD 분석: {macd_signal} ({macd_strength})")

        # 피보나치 분석
        fib_signal = "매수" if analysis_results['fibonacci']['signal'] == 'Buy' else "매도" if analysis_results['fibonacci']['signal'] == 'Sell' else "중립"
        fib_strength = "강함" if analysis_results['fibonacci']['strength'] == 'Strong' else "약함" if analysis_results['fibonacci']['strength'] == 'Weak' else "중립"
        st.write(f"- 피보나치 분석: {fib_signal} ({fib_strength})")

        # Volume Super Trend AI 분석
        st.write("Volume Super Trend AI 분석:")
        vsta = analysis_results['volume_super_trend_ai']
        st.write(f"- 신호: {'매수' if vsta['signal'] == 'Buy' else '매도'}")
        st.write(f"- 강도: {vsta['strength']}")
        st.write(f"- Super Trend 값: {vsta['super_trend_value']:.2f}")
        st.write(f"- 추세 강도: {vsta['trend_strength']:.2f}")

        # 급락 후 반등 분석
        st.write("급락 후 반등 분석:")
        rfr = analysis_results['rapid_fall_rebound']
        st.write(f"- 신호: {rfr['signal']}")
        st.write(f"- 강도: {rfr['strength']}")
        st.write(f"- 이유: {rfr['reason']}")

        # 전체 신호 표시 부분 수정
        overall_signal = analysis_results['overall_signal']
        st.subheader("종합 분석 결과")
        st.write(f"전체 신호: {'매수' if overall_signal['signal'] == 'Buy' else '매도' if overall_signal['signal'] == 'Sell' else '중립'}")
        st.write(f"신호 강도: {overall_signal['strength']}")


# 누적 예측 정확도 표시
    st.subheader('누적 예측 정확도 (최근 3일)')
    if 'cumulative_predictions' in coin_prediction:
        cumulative_preds = coin_prediction['cumulative_predictions']
        if cumulative_preds:
            data = []
            for pred in cumulative_preds[-72:]:  # 최근 3일 데이터 (1시간 간격으로 72개)
                data.append({
                    '예측 시간': pred['prediction_time'],
                    '현재 가격': pred['current_price'],
                    '30분 후 예측': pred['predicted_price_30min'],
                    '30분 후 오차': pred['error_30min'],
                    '30분 후 오차율': f"{pred['error_percentage_30min']:.2f}%",
                    '1시간 후 예측': pred['predicted_price_1hour'],
                    '1시간 후 오차': pred['error_1hour'],
                    '1시간 후 오차율': f"{pred['error_percentage_1hour']:.2f}%",
                    '24시간 후 예측': pred['predicted_price_24hours'],
                    '24시간 후 오차': pred['error_24hours'],
                    '24시간 후 오차율': f"{pred['error_percentage_24hours']:.2f}%"
                })
            df = pd.DataFrame(data)
            st.dataframe(df)
        
        # 오차율 추이 그래프
        st.subheader('오차율 추이')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['예측 시간'], y=df['30분 후 오차율'], mode='lines', name='30분 후'))
        fig.add_trace(go.Scatter(x=df['예측 시간'], y=df['1시간 후 오차율'], mode='lines', name='1시간 후'))
        fig.add_trace(go.Scatter(x=df['예측 시간'], y=df['24시간 후 오차율'], mode='lines', name='24시간 후'))
        fig.update_layout(title='예측 시간별 오차율 추이', xaxis_title='예측 시간', yaxis_title='오차율 (%)')
        st.plotly_chart(fig)
    else:
        st.write("누적 예측 데이터가 없습니다.")

    # 차트 표시
    st.subheader('가격 차트 및 기술적 지표')
    chart = create_technical_chart(coin_data)
    st.plotly_chart(chart, use_container_width=True)

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

if __name__ == '__main__':
    main()
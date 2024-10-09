import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data_collector import collect_data, collect_upbit_data
from technical_analysis import perform_analysis
from price_predictor import predict_prices
from datetime import datetime, timedelta

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

    # 볼린저 밴드 (4시간, 4 표준편차)
    fig.add_trace(go.Scatter(x=df.index, y=df['upper_band_4'], name='상단 밴드 (4, 4)',
                             line=dict(color='rgba(0, 0, 255, 0.3)')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA4'], name='4시간 이동평균',
                             line=dict(color='rgba(0, 0, 255, 0.8)')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['lower_band_4'], name='하단 밴드 (4, 4)',
                             line=dict(color='rgba(0, 0, 255, 0.3)')), row=1, col=1)

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

    # 사이드바에서 코인 선택 (radio 버튼 사용)
    coin_options = ['BTC', 'ETH', 'XRP', 'SOL', 'SUI']
    selected_coin = st.sidebar.radio('코인 선택', coin_options)

    st.write(f'선택된 코인: {selected_coin}')

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
        predictions = predict_prices(selected_coin, coin_data, analysis_results, current_price, collected_data['news_data'].get(selected_coin, []), collected_data.get('economic_data', []))

        # 가격 예측 결과 표시 (상단으로 이동)
        st.subheader('가격 예측')
        st.write(f"현재 가격: {int(current_price):,}원 (예측 시간: {timestamp})")
        col1, col2, col3 = st.columns(3)
        col1.metric("30분 후 예상 가격", f"{int(predictions['predicted_price_30min']):,}원", 
                    f"{int(predictions['predicted_price_30min'] - current_price):,}원")
        col2.metric("1시간 후 예상 가격", f"{int(predictions['predicted_price_1hour']):,}원", 
                    f"{int(predictions['predicted_price_1hour'] - current_price):,}원")
        col3.metric("24시간 후 예상 가격", f"{int(predictions['predicted_price_24hours']):,}원", 
                    f"{int(predictions['predicted_price_24hours'] - current_price):,}원")
        
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

        # 예측 정확도 표시 (30분, 1시간, 24시간 전 예측 정확도 계산)
        if 'error_30min' in predictions and 'error_1hour' in predictions and 'error_24hours' in predictions:
            st.subheader('이전 예측 정확도')
            st.write(f"30분 전 오차: {int(predictions['error_30min']['error']):,}원, 오차율: {predictions['error_30min']['error_percentage']:.2f}%")
            st.write(f"1시간 전 오차: {int(predictions['error_1hour']['error']):,}원, 오차율: {predictions['error_1hour']['error_percentage']:.2f}%")
            st.write(f"24시간 전 오차: {int(predictions['error_24hours']['error']):,}원, 오차율: {predictions['error_24hours']['error_percentage']:.2f}%")
            st.write("오차 감소 여부:")
            st.write(f"- 30분 전 -> 1시간 전: {'감소' if predictions['error_1hour']['error'] < predictions['error_30min']['error'] else '증가'}")
            st.write(f"- 1시간 전 -> 24시간 전: {'감소' if predictions['error_24hours']['error'] < predictions['error_1hour']['error'] else '증가'}")

        # 누적 예측 정확도 표시 (최근 3일까지의 데이터)
        if 'cumulative_errors' in predictions:
            st.subheader('누적 예측 정확도 (최근 3일)')
            cumulative_errors = predictions['cumulative_errors'][-3:]  # 최근 3일까지의 데이터만 표시
            if cumulative_errors:
                cumulative_df = pd.DataFrame(cumulative_errors)
                cumulative_df['timestamp'] = cumulative_df['timestamp'].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
                st.table(cumulative_df[['timestamp', 'predicted_30min', 'error_30min', 'error_percentage_30min',
                                        'predicted_1hour', 'error_1hour', 'error_percentage_1hour',
                                        'predicted_24hours', 'error_24hours', 'error_percentage_24hours']])
            else:
                st.write("누적 데이터가 없습니다.")

    # 차트 표시
    st.subheader('가격 차트 및 기술적 지표')
    chart = create_technical_chart(coin_data)
    st.plotly_chart(chart, use_container_width=True)

if __name__ == '__main__':
    main()
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 데이터 파일 경로
DATA_PATH = 'data'
HISTORY_FILE = os.path.join(DATA_PATH, 'prediction_history.json')

# 코인별 텍스트 색상 정의
COIN_COLORS = {
    'KRW-BTC': '#3498db',  # 파란색
    'KRW-ETH': '#e74c3c',  # 빨간색
    'KRW-SOL': '#2ecc71',  # 초록색
    'KRW-SUI': '#9b59b6',  # 보라색
}

def load_data():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
            return history
        except json.JSONDecodeError:
            logging.error(f"JSON 디코딩 오류: {HISTORY_FILE}")
        except Exception as e:
            logging.error(f"데이터 로드 중 오류 발생: {e}")
    logging.warning(f"데이터 파일을 찾을 수 없음: {HISTORY_FILE}")
    return []

def calculate_prediction_accuracy(predicted, actual):
    if predicted is None or actual is None or predicted == 0:
        return None
    try:
        return (actual - predicted) / predicted * 100
    except Exception as e:
        logging.error(f"정확도 계산 중 오류 발생: {e}")
        return None

def calculate_price_difference(predicted, actual):
    if predicted is None or actual is None:
        return 'N/A'
    difference = int(actual) - int(predicted)  # 정수로 변환
    sign = '+' if difference >= 0 else '-'
    return f"{sign}{abs(difference):,}"

def predictions_to_dataframe(predictions, is_cumulative=False):
    data = []
    for p in predictions:
        try:
            current_price = int(p['current_price'])
            predicted_price_1h = int(p['predicted_price_1h'])
            predicted_price_24h = int(p['predicted_price_24h'])
            
            previous_prediction_1h = p.get('previous_prediction_1h')
            previous_prediction_24h = p.get('previous_prediction_24h')
            
            accuracy_1h = p.get('accuracy_1h')
            accuracy_24h = p.get('accuracy_24h')

            # 1시간 예측 차이 계산
            diff_1h = calculate_price_difference(previous_prediction_1h, current_price)
            
            # 24시간 예측 차이 계산 (누적 데이터에서만)
            if is_cumulative and previous_prediction_24h is not None:
                diff_24h = calculate_price_difference(previous_prediction_24h, current_price)
            else:
                diff_24h = 'N/A'

            data.append({
                '예측 시간': datetime.fromisoformat(p['timestamp']).strftime('%Y-%m-%d %H:%M:%S'),
                '코인': p['coin'],
                '현재 가격': f"{current_price:,}",
                '1시간 전 예측': f"{int(previous_prediction_1h):,}" if previous_prediction_1h is not None else 'N/A',
                '1시간 예측 차이': diff_1h,
                '1시간 예측 정확도(%)': f"{accuracy_1h:.2f}%" if accuracy_1h is not None else 'N/A',
                '1시간 후 예측': f"{predicted_price_1h:,}",
                '24시간 전 예측': f"{int(previous_prediction_24h):,}" if previous_prediction_24h is not None and is_cumulative else 'N/A',
                '24시간 예측 차이': diff_24h,
                '24시간 예측 정확도(%)': f"{accuracy_24h:.2f}%" if accuracy_24h is not None and is_cumulative else 'N/A',
                '24시간 후 예측': f"{predicted_price_24h:,}",
                '1시간 추세': '↑' if p['price_trend_1h'] == '상승 추세' else '↓',
                '24시간 추세': '↑' if p['price_trend_24h'] == '상승 추세' else '↓',
            })
        except Exception as e:
            logging.error(f"데이터 처리 중 오류 발생: {e}")
    return pd.DataFrame(data)

def df_to_html_table(df):
    table_html = '<table style="width:100%; border-collapse: collapse; color: white;">'
    table_html += '<tr>'
    for col in df.columns:
        table_html += f'<th style="border: 1px solid #444; padding: 8px; text-align: center; background-color: #222;">{col}</th>'
    table_html += '</tr>'
    
    for _, row in df.iterrows():
        table_html += '<tr>'
        for col in df.columns:
            value = row[col]
            if col == '코인':
                color = COIN_COLORS.get(value, 'white')
                table_html += f'<td style="border: 1px solid #444; padding: 8px; text-align: center; color: {color};">{value}</td>'
            elif col in ['1시간 추세', '24시간 추세']:
                color = 'lightblue' if value == '↑' else 'lightcoral'
                table_html += f'<td style="border: 1px solid #444; padding: 8px; text-align: center; color: {color};">{value}</td>'
            elif col in ['1시간 예측 차이', '24시간 예측 차이']:
                if value != 'N/A':
                    color = 'lightgreen' if value.startswith('+') else 'lightcoral'
                    table_html += f'<td style="border: 1px solid #444; padding: 8px; text-align: center; color: {color};">{value}</td>'
                else:
                    table_html += f'<td style="border: 1px solid #444; padding: 8px; text-align: center;">{value}</td>'
            else:
                table_html += f'<td style="border: 1px solid #444; padding: 8px; text-align: center;">{value}</td>'
        table_html += '</tr>'
    
    table_html += '</table>'
    return table_html

def main():
    st.set_page_config(layout="wide", page_title="암호화폐 가격 예측 대시보드")
    
    # 다크 테마 적용
    st.markdown("""
    <style>
    .reportview-container {
        background-color: #0e1117;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #262730;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title('암호화폐 가격 예측 대시보드')

    # 데이터 로드
    history = load_data()

    if not history:
        st.write("아직 예측 데이터가 없습니다. 첫 예측이 완료되면 여기에 표시됩니다.")
        return

    # 최신 예측 결과 표시
    st.header("최신 예측 결과")
    if history and isinstance(history[-1], list) and history[-1]:
        try:
            latest_df = predictions_to_dataframe(history[-1])
            st.markdown(df_to_html_table(latest_df), unsafe_allow_html=True)
        except Exception as e:
            st.error(f"최신 예측 데이터 처리 중 오류 발생: {e}")
    else:
        st.write("최신 예측 데이터를 표시할 수 없습니다.")

    # 누적 데이터 표시 (별도 섹션)
    st.header("누적 예측 데이터 (최근 48시간)")
    cut_off_time = datetime.now() - timedelta(hours=48)
    cumulative_predictions = []
    for predictions in reversed(history):
        if predictions and isinstance(predictions, list):
            for p in predictions:
                try:
                    if datetime.fromisoformat(p['timestamp']) > cut_off_time:
                        cumulative_predictions.append(p)
                except (KeyError, ValueError) as e:
                    logging.error(f"누적 데이터 처리 중 오류 발생: {e}")

    if cumulative_predictions:
        try:
            cumulative_df = predictions_to_dataframe(cumulative_predictions, is_cumulative=True)
            st.markdown(df_to_html_table(cumulative_df), unsafe_allow_html=True)
        except Exception as e:
            st.error(f"누적 예측 데이터 처리 중 오류 발생: {e}")
    else:
        st.write("표시할 누적 예측 데이터가 없습니다.")

if __name__ == "__main__":
    main()
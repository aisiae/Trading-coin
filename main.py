import os
import schedule
import time
import logging
from datetime import datetime
from data_collector import collect_upbit_data, get_current_price, collect_data
from technical_analysis import perform_analysis
from price_predictor import predict_prices
import json

# 파일과 콘솔 모두에 로그를 출력하도록 설정
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("coin_prediction.log"),
                        logging.StreamHandler()
                    ])

COINS = ['BTC', 'ETH', 'XRP', 'SOL', 'SUI']

def save_predictions(predictions):
    previous_predictions = load_previous_predictions()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 새로운 예측 데이터와 이전 데이터 병합
    for coin, pred in predictions.items():
        if coin not in previous_predictions:
            previous_predictions[coin] = []
        elif not isinstance(previous_predictions[coin], list):
            previous_predictions[coin] = [previous_predictions[coin]]
        
        # 새 예측에 타임스탬프 추가
        pred['timestamp'] = timestamp
        
        # 최근 10개의 예측만 유지
        previous_predictions[coin] = previous_predictions[coin][-9:] + [pred]
    
    filename = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(filename, 'w') as f:
        json.dump(previous_predictions, f, indent=4)

    logging.info(f"Predictions saved to {filename}")


def load_previous_predictions():
    files = [f for f in os.listdir() if f.startswith("predictions_") and f.endswith(".json")]
    if not files:
        return {}
    latest_file = max(files)
    with open(latest_file, 'r') as f:
        try:
            data = json.load(f)
            # 각 코인의 예측 데이터가 리스트가 아니면 리스트로 변환
            for coin in data:
                if not isinstance(data[coin], list):
                    data[coin] = [data[coin]]
            return data
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from {latest_file}")
            return {}

def save_predictions(predictions):
    previous_predictions = load_previous_predictions()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # predictions 디렉토리 생성 (없는 경우)
    if not os.path.exists('predictions'):
        os.makedirs('predictions')
    
    # 새로운 예측 데이터와 이전 데이터 병합
    for coin, pred in predictions.items():
        if coin not in previous_predictions:
            previous_predictions[coin] = []
        elif not isinstance(previous_predictions[coin], list):
            previous_predictions[coin] = [previous_predictions[coin]]
        
        # 새 예측에 타임스탬프 추가
        pred['timestamp'] = timestamp
        
        # 최근 10개의 예측만 유지
        previous_predictions[coin] = previous_predictions[coin][-9:] + [pred]
    
    filename = os.path.join('predictions', f"predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
    with open(filename, 'w') as f:
        json.dump(previous_predictions, f, indent=4)

    logging.info(f"Predictions saved to {filename}")

def job():
    try:
        logging.info("작업 시작")
        
        # 데이터 수집
        collected_data = collect_data()
        all_predictions = {}
        
        for coin in COINS:
            logging.info(f"{coin} 처리 중...")
            
            # 현재 가격 가져오기
            current_price = get_current_price(coin)
            if current_price is None:
                logging.warning(f"{coin}의 현재 가격을 가져오는데 실패했습니다")
                continue
            
            logging.info(f"{coin}의 현재 가격: {current_price}")
            
            # 코인별 데이터 수집
            coin_data = collect_upbit_data(coin)

            if coin_data.empty:
                logging.warning(f"{coin}의 데이터 수집에 실패했습니다")
                continue

            # 기술적 분석 수행
            analysis_results = perform_analysis(coin_data)

            # 뉴스 데이터 가져오기
            news_data = collected_data.get('us_economic_news', [])

            # 가격 예측
            economic_data = collected_data.get('economic_data', {})
            predictions = predict_prices(coin, coin_data, analysis_results, current_price, news_data, economic_data)

            # 예측 결과 저장
            if predictions:
                predictions['current_price'] = current_price
                predictions['prediction_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                all_predictions[coin] = predictions

                logging.info(f"{coin} 예측 결과:")
                logging.info(f"현재 가격: {current_price}")
                logging.info(f"30분 후 예상 가격: {predictions.get('predicted_price_30min', 'N/A')}")
                logging.info(f"1시간 후 예상 가격: {predictions.get('predicted_price_1hour', 'N/A')}")
                logging.info(f"24시간 후 예상 가격: {predictions.get('predicted_price_24hours', 'N/A')}")
                logging.info(f"예측 이유: {predictions.get('reason', '이유 없음')}")
    
                # 오차 정보 로깅
                if 'last_error' in predictions:
                    logging.info(f"마지막 예측 오차: {predictions['last_error']}")
                    logging.info(f"마지막 예측 오차 비율: {predictions['last_error_percentage']}%")
            else:
                logging.warning(f"{coin} 예측 실패")
            
            logging.info("-" * 50)
        
        # 모든 예측 결과 저장
        save_predictions(all_predictions)
        
        logging.info("작업 완료")
    except Exception as e:
        logging.error(f"작업 중 오류 발생: {str(e)}", exc_info=True)

        for coin in COINS:
            predictions_file = os.path.join('predictions', f'predictions_{coin}.csv')
        if os.path.exists(predictions_file):
            predictions_df = pd.read_csv(predictions_file, parse_dates=['timestamp'])
            print(f"\n최근 {coin} 예측 데이터:")
            print(predictions_df.tail())
        else:
            print(f"{coin}에 대한 예측 데이터가 없습니다.")

def run_schedule():
    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except Exception as e:
            logging.error(f"스케줄러 실행 중 오류 발생: {str(e)}", exc_info=True)
            time.sleep(60)  # 오류 발생 시 1분 대기 후 재시도

def main():
    logging.info("코인 가격 예측 시스템 시작")

    # 즉시 한 번 실행
    job()

    # 30분마다 job 함수 실행 예약
    schedule.every(30).minutes.do(job)

    # 스케줄러 실행
    run_schedule()

if __name__ == "__main__":
    main()
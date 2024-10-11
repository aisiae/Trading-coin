import schedule
import time
import logging
from datetime import datetime
from data_collector import collect_upbit_data, get_current_price, collect_data
from technical_analysis import perform_analysis
from price_predictor import predict_prices
import logging
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"predictions_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(predictions, f)

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
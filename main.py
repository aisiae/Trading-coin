import schedule
import time
import logging
from datetime import datetime
from data_collector import collect_upbit_data, get_current_price, collect_data
from technical_analysis import perform_analysis
from price_predictor import predict_prices

logging.basicConfig(filename='coin_prediction.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

COINS = ['BTC', 'ETH', 'XRP', 'SOL', 'SUI']

def job():
    try:
        logging.info("작업 시작")
        
        # 데이터 수집
        collected_data = collect_data()
        
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
            news_data = collected_data.get('news_data', {}).get(coin, [])

            # 가격 예측
            economic_data = collected_data.get('economic_data', [])
            predictions = predict_prices(coin, coin_data, analysis_results, current_price, news_data, economic_data)

            # 결과 출력
            if predictions:
                logging.info(f"{coin} 예측:")
                logging.info(predictions)
            else:
                logging.warning(f"{coin} 예측 실패")
            
            logging.info("-" * 50)

        logging.info("작업 완료")
    except Exception as e:
        logging.error(f"작업 중 오류 발생: {str(e)}", exc_info=True)

def main():
    logging.info("코인 가격 예측 시스템 시작")

    # 즉시 한 번 실행
    job()

    # 30분마다 job 함수 실행 예약
    schedule.every(30).minutes.do(job)

    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except Exception as e:
            logging.error(f"스케줄러 실행 중 오류 발생: {str(e)}", exc_info=True)
            time.sleep(60)  # 오류 발생 시 1분 대기 후 재시도

if __name__ == "__main__":
    main()
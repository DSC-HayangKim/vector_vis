from database import MongoDatabase
import requests
import xmltodict
import time
import pymongo


def fetch_and_insert_tech_data(page: int = 1, list_count: int = 20):
    # 1. 기본 URL (나중에 진짜 주소로 변경 시 여기만 수정)
    base_url = "https://tb.kibo.or.kr/ktbs/voc/openapi/openApiCall.do"

    # 2. 파라미터 분리 (가독성 및 유지보수 향상)
    params = {
        "key": "Ov5QZCcLNiBScGCA23o9S646XlHLZrmkXnMqKwPDmOuV4KOtPYbz4ewFCQMk",
        "aseq": 2,
        "page": page,  # 페이지 자동 변경
        "listCount": list_count
    }

    print(f" API 호출 중... (Page: {page})")

    try:
        # requests가 params를 URL 뒤에 자동으로 붙여줌
        response = requests.get(base_url, params=params)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f" API 호출 오류: {e}")
        return

    # 3. XML 파싱
    try:
        xml_dict: dict = xmltodict.parse(response.text)
    except Exception as e:
        print(f" XML 파싱 오류: {e}")
        return

    # 4. 데이터 추출
    items = None
    try:
        body = xml_dict.get('response', {}).get('body', {})
        items_wrapper = body.get('items', {})
        items = items_wrapper.get('item')
    except AttributeError:
        print("⚠ XML 구조가 예상과 다릅니다.")
        return

    if not items:
        print(f"️ Page {page}: 저장할 데이터가 없습니다.")
        return

    # 리스트가 아닌 단일 딕셔너리로 올 경우 리스트로 변환
    if isinstance(items, dict):
        items = [items]

    # 5. MongoDB 저장
    mongo_db = MongoDatabase(database_name="crawler_db")
    collection_name = "HopeTech"

    print(f" 총 {len(items)}개의 문서를 저장합니다...")

    for idx, item_document in enumerate(items):
        try:
            # 크롤링 메타데이터 추가 (선택)
            item_document['source_page'] = page

            mongo_db.insert_document(collection_name, item_document)
            # print(f" - [{idx+1}] 저장 완료") # 너무 시끄러우면 주석 처리

        except pymongo.errors.PyMongoError as e:
            print(f" DB 저장 오류: {e}")

    print(" 페이지 저장 완료")

    # 연결 종료 (리소스 관리)
    if mongo_db.client:
        mongo_db.client.close()


if __name__ == "__main__":
    start_page = 1  # 시작 페이지 설정
    end_page = 100  # 끝 페이지 설정 (필요시 10000 등으로 수정)

    try:
        for i in range(start_page, end_page + 1):
            fetch_and_insert_tech_data(page=i, list_count=20)
            time.sleep(3)  # 서버 부하 방지를 위해 3~11초 대기
    except KeyboardInterrupt:
        print("\n 사용자에 의해 중단되었습니다.")
    except Exception as error:
        print(f" 치명적 오류 발생: {error}")
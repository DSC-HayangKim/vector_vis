import os
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient

# 1. 환경 변수 로드
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# 2. 연결 정보 가져오기
USER = os.getenv("DB_USER")
PW = os.getenv("DB_PASSWORD")
HOST = os.getenv("DB_HOST")  # localhost여야 함
PORT = os.getenv("DB_PORT")  # 20060여야 함

# 3. 접속 시도
uri = f"mongodb://{USER}:{PW}@{HOST}:{PORT}/"
print(f"📡 접속 시도: mongodb://{USER}:****@{HOST}:{PORT}/")

try:
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)

    # 4. DB 목록 확인
    dbs = client.list_database_names()
    print(f"\n📂 존재하는 데이터베이스 목록: {dbs}")

    if 'crawler_db' not in dbs:
        print("🚨 [문제 발견] 'crawler_db'가 목록에 없습니다! 크롤러가 저장을 안 했거나 DB 이름이 다릅니다.")
    else:
        db = client['crawler_db']

        # 5. 컬렉션 목록 확인
        cols = db.list_collection_names()
        print(f"   └─ 'crawler_db' 안의 컬렉션들: {cols}")

        if 'HopeTech' in cols:
            count = db['HopeTech'].count_documents({})
            print(f"\n📊 'HopeTech' 데이터 개수: {count}개")

            if count > 0:
                # 6. 실제 데이터 1개 꺼내서 키값(Key) 확인
                sample = db['HopeTech'].find_one()
                print("\n🔎 [중요] 첫 번째 데이터 샘플 (Key 확인용):")
                print("------------------------------------------------")
                print(sample)  # 여기서 실제 키값이 'techName'인지 확인해야 함
                print("------------------------------------------------")
            else:
                print("🚨 [문제 발견] 컬렉션은 있는데 데이터가 0개입니다.")
        else:
            print("🚨 [문제 발견] 'HopeTech' 컬렉션이 없습니다.")

except Exception as e:
    print(f"❌ 접속 실패 에러: {e}")
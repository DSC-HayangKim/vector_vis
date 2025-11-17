import os
import platform
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from konlpy.tag import Okt

# UserWarning 무시
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class PatentAnalyzer:
    def __init__(self):
        # 환경 변수 자동 로드 (.env 파일 경로 탐색)
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            print(f".env 환경 변수 로드 완료: {env_path}")
        else:
            print(".env 파일을 찾을 수 없습니다. 환경 변수 확인 요망.")

        # OS별 한글 폰트 적용
        self._set_korean_font()

        # MongoDB 접속 정보 로드 (.env)
        self.db_user = os.getenv("DB_USER")
        self.db_password = os.getenv("DB_PASSWORD")
        self.db_host = os.getenv("DB_HOST")  # ex) localhost
        self.db_port = os.getenv("DB_PORT")  # ex) 20060

        # (참고) SSH 터널 사용 시, 접속 주소 (srv 제거, 포트 명시)
        self.mongo_uri = f"mongodb://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/"

        # 접속 설정 출력 (비밀번호는 숨김 처리)
        print(f"MongoDB URI: mongodb://{self.db_user}:****@{self.db_host}:{self.db_port}/")

    def _set_korean_font(self):
        """OS에 맞는 한글 폰트 설정"""
        system_name = platform.system()
        if system_name == 'Windows':
            plt.rc('font', family='Malgun Gothic')
        elif system_name == 'Darwin':  # Mac
            plt.rc('font', family='AppleGothic')
        else:
            plt.rc('font', family='NanumGothic')
        plt.rcParams['axes.unicode_minus'] = False

    def load_data(self, db_name='crawler_db', col_name='HopeTech'):
        """MongoDB에서 특허 데이터 로드 (필요 필드만, 샘플링 적용)"""
        print(f"MongoDB({db_name}.{col_name}) 연결 중...")
        try:
            client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
            db = client[db_name]
            collection = db[col_name]

            # 성능 테스트용, 최신 n건만 가져옴 (limit 해제시 전체 분석)
            limit_count = 100_000
            print(f"데이터 로딩 ({limit_count}건 샘플) ...")

            cursor = collection.find({}, {'_id': 0, 'techname': 1, 'tecactr': 1}).limit(limit_count)
            data = list(cursor)

            if not data:
                print("DB에 데이터가 없습니다. 샘플 데이터로 대체합니다.")
                return self._get_sample_data()

            df = pd.DataFrame(data)
            print(f"데이터 로드 완료: {len(df)} rows")

            # 결측값 제거
            return df.dropna(subset=['techname', 'tecactr'])

        except Exception as e:
            print(f"DB 로드/연결 오류: {e}")
            print("SSH 터널 상태 및 접속 정보 재확인 요망.")
            return self._get_sample_data()

    def _get_sample_data(self):
        """DB 연결 실패 시 임시 샘플 데이터 반환"""
        return pd.DataFrame({
            'techname': ['AI 이미지 분석', '자율주행 센서', '블록체인 보안', '딥러닝 의료'],
            'tecactr': ['인공지능 신경망...', '라이다 센서...', '분산원장 기술...', 'MRI 분석...']
        })

    def preprocess_text(self, df):
        """텍스트 정제: 형태소 분석 후 불용어 제거"""
        print("텍스트 전처리 및 형태소 분석 중...")
        okt = Okt()
        # 특허 불용어 리스트 정의
        stop_words = ['및', '를', '을', '이', '가', '본', '기술', '제공', '방법', '장치', '시스템', '의', '에', '하여', '구비', '포함', '특징',
                      '형성', '구성', '단계']

        def clean(text):
            if pd.isna(text): return ""
            text = str(text)
            nouns = okt.nouns(text)
            return " ".join([n for n in nouns if n not in stop_words and len(n) > 1])

        # techname/tecactr 결합 후 분석 정확도 향상
        df['full_text'] = df['techname'] + " " + df['tecactr']
        df['processed_text'] = df['full_text'].apply(clean)

        # 빈 텍스트 행 제거
        df = df[df['processed_text'].str.strip() != ""]
        return df

    def find_optimal_k(self, vectors, max_k=10):
        """엘보우 기법 기반 최적 군집 수(K) 산출"""
        print("엘보우 기법으로 최적 군집 수 탐색")
        inertias = []
        k_range = range(1, min(max_k, vectors.shape[0]) + 1)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(vectors)
            inertias.append(kmeans.inertia_)

        # Elbow 포인트 계산
        if len(k_range) > 2:
            x1, y1 = k_range[0], inertias[0]
            x2, y2 = k_range[-1], inertias[-1]
            distances = []
            for i, k in enumerate(k_range):
                x0, y0 = k, inertias[i]
                numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
                denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
                distances.append(numerator / denominator)
            optimal_k = k_range[np.argmax(distances)]
        else:
            optimal_k = 1

        print(f"최적 군집 수: {optimal_k}")
        # 시각화 저장
        plt.figure(figsize=(8, 4))
        plt.plot(k_range, inertias, 'bo-')
        plt.plot(optimal_k, inertias[optimal_k - 1], 'ro', label='Optimal K')
        plt.title(f'Elbow Method (Optimal k={optimal_k})')
        plt.legend()
        plt.grid(True)
        plt.savefig('elbow_chart.png')
        plt.close()

        return optimal_k

    def analyze_and_visualize(self, df):
        if df.empty:
            print("분석 가능한 데이터가 없습니다.")
            return

        # TF-IDF 벡터화
        print("TF-IDF 벡터화 진행 중...")
        tfidf = TfidfVectorizer(max_features=1000)
        vectors = tfidf.fit_transform(df['processed_text'])

        # 최적 클러스터 수 자동 탐색
        if vectors.shape[0] < 3:
            optimal_k = 1
        else:
            optimal_k = self.find_optimal_k(vectors)

        # K-Means 군집화
        print(f"K-Means 군집화 (k={optimal_k}) 수행 중...")
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(vectors)

        # t-SNE 좌표 생성
        print("t-SNE 임베딩 중...")
        n_samples = vectors.shape[0]
        perp = min(30, n_samples - 1) if n_samples > 1 else 1

        tsne = TSNE(n_components=2, random_state=42, perplexity=perp, init='pca', learning_rate='auto')
        tsne_results = tsne.fit_transform(vectors.toarray())

        df['x'] = tsne_results[:, 0]
        df['y'] = tsne_results[:, 1]

        self._draw_plot(df, optimal_k)

    def _draw_plot(self, df, k_count):
        plt.figure(figsize=(12, 8))

        # 군집 별 색상 팔레트 자동 설정
        palette = sns.color_palette("viridis", as_cmap=False, n_colors=k_count)

        sns.scatterplot(x='x', y='y', hue='cluster', palette=palette,
                        data=df, s=100, alpha=0.8, legend='full')

        # 최대 30개 라벨, 길면 자름
        limit = min(30, len(df))
        for i in range(limit):
            label = str(df['techname'].iloc[i])
            if len(label) > 10: label = label[:10] + "..."
            plt.text(df['x'].iloc[i], df['y'].iloc[i] + 0.2, label, fontsize=9)

        plt.title(f'Patent Tech Trend Map (Total Clusters: {k_count})')
        plt.grid(True, alpha=0.3)
        plt.savefig('patent_analysis_result.png', dpi=300, bbox_inches='tight')
        print("시각화 결과는 'patent_analysis_result.png'에서 확인 가능합니다.")
        plt.show()


if __name__ == "__main__":
    analyzer = PatentAnalyzer()

    # 1. 데이터 로드
    df = analyzer.load_data()

    # 2. 전처리
    df = analyzer.preprocess_text(df)

    # 3. 분석 및 시각화 (자동)
    analyzer.analyze_and_visualize(df)

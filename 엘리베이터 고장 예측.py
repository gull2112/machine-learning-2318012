import pandas as pd  # 데이터 조작 및 분석을 위한 라이브러리
import numpy as np  # 수치 연산을 위한 라이브러리
import seaborn as sns  # 데이터 시각화를 위한 라이브러리
import matplotlib.pyplot as plt  # 그래프를 그리기 위한 라이브러리
import os  # 파일 및 디렉토리 관련 작업을 위한 라이브러리
from sklearn.model_selection import train_test_split  # 데이터셋을 학습/테스트로 분리
from sklearn.ensemble import RandomForestClassifier  # 랜덤 포레스트 모델 (분류)
from sklearn.impute import SimpleImputer  # 결측값 처리
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix  # 모델 평가 지표
from sklearn.preprocessing import StandardScaler  # 데이터 표준화

# 결과를 저장할 디렉토리 생성
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)  # 디렉토리가 없으면 생성

# 엑셀 파일에서 데이터셋 로드
file_path = './data/2.elevator_failure_prediction.xlsx'
data = pd.ExcelFile(file_path)
df = data.parse('data')  # 'data'라는 이름의 시트를 파싱하여 데이터프레임으로 변환

# 데이터셋 요약 및 결측값 정보를 텍스트 파일로 저장
with open(os.path.join(results_dir, 'dataset_summary.txt'), 'w') as f:
    f.write("Dataset Summary:\n")
    f.write(str(df.describe()))  # 데이터셋 요약 통계
    f.write("\n\nMissing Values:\n")
    f.write(str(df.isnull().sum()))  # 각 열의 결측값 개수

# 상관 관계 행렬을 히트맵으로 생성 및 저장
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')  # 상관 관계를 히트맵으로 표현
plt.title('Correlation Matrix')
plt.savefig(os.path.join(results_dir, 'correlation_matrix.png'))  # 이미지를 파일로 저장
plt.close()

# 'Time' 열 제거 (모델링에 불필요한 열로 가정)
df = df.drop(columns=['Time'])

# 결측값을 평균값으로 대체
imputer = SimpleImputer(strategy='mean')
X = df.drop(columns=['Status'])  # 독립 변수 (입력 데이터)
y = df['Status']  # 종속 변수 (타겟 데이터)
X_imputed = imputer.fit_transform(X)

# 데이터를 표준화하여 스케일 조정
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# 데이터를 학습/테스트 세트로 분리 (20%는 테스트 데이터로 사용)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 랜덤 포레스트 모델 학습
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 테스트 데이터에 대해 예측 수행
y_pred = model.predict(X_test)

# 모델 평가
accuracy = accuracy_score(y_test, y_pred)  # 정확도 계산
report = classification_report(y_test, y_pred)  # 분류 보고서 생성
conf_matrix = confusion_matrix(y_test, y_pred)  # 혼동 행렬 생성

# 평가 결과를 텍스트 파일로 저장
with open(os.path.join(results_dir, 'evaluation_results.txt'), 'w') as f:
    f.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")  # 정확도 기록
    f.write("Classification Report:\n")
    f.write(report)  # 분류 보고서 기록

# 혼동 행렬 히트맵 생성 및 저장
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y), yticklabels=np.unique(y))  # 혼동 행렬 히트맵
plt.title('Confusion Matrix')
plt.xlabel('Predicted')  # 예측값
plt.ylabel('Actual')  # 실제값
plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
plt.close()

# 각 피처의 중요도를 계산하고 시각화
feature_importances = model.feature_importances_  # 피처 중요도 계산
feature_names = df.drop(columns=['Status']).columns  # 피처 이름 가져오기

plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importances, align='center')  # 수평 바 차트로 중요도 표현
plt.xlabel('Feature Importance')  # 중요도 축 라벨
plt.title('Random Forest Feature Importance')  # 그래프 제목
plt.savefig(os.path.join(results_dir, 'feature_importance.png'))  # 그래프 저장
plt.close()

# 타겟 변수 분포 시각화
plt.figure(figsize=(10, 6))
sns.countplot(x=y)  # 타겟 변수의 카운트 플롯
plt.title('Target Variable Distribution')  # 그래프 제목
plt.xlabel('Status')  # x축 라벨
plt.ylabel('Count')  # y축 라벨
plt.savefig(os.path.join(results_dir, 'target_distribution.png'))  # 그래프 저장
plt.close()

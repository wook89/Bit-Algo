# 📊 Bitalgo

실시간 가상자산 분석을 위한 **Streamlit 기반 웹 애플리케이션**입니다.  
차트 시각화, 기술적 분석 지표, 인터랙티브 UI를 통해 간편하게 시장 동향을 확인할 수 있습니다.  

---

## 🚀 데모

👉 [Bitalgo 바로가기](https://bitalgo.streamlit.app/)  



---

## ✨ 주요 기능

- 📈 **실시간 시세 확인**: 주요 코인 가격 및 변동률 조회  
- 📊 **차트 시각화**: 캔들차트, 이동평균선 등 다양한 기술적 지표 표시  
- ⚡ **사용자 입력 기반 분석**: 기간/코인 선택 후 맞춤형 분석 제공  
- 📉 **위험 관리 도구**: 손절/익절 지점 설정 및 시뮬레이션  
- 🌐 **웹 앱 배포**: Streamlit Cloud 환경에서 누구나 접속 가능  

---

## 🛠 기술 스택

- **Frontend / UI**: [Streamlit](https://streamlit.io/)  
- **Backend / Data**: Python (Pandas, NumPy)  
- **Visualization**: Plotly, Matplotlib  
- **API 연동**: 가상자산 시세 데이터 API (예: Binance, CoinGecko)  

---

## ⚙️ 설치 방법 (로컬 실행)

```bash
# 1. 저장소 클론
git clone https://github.com/사용자명/Bitalgo.git
cd Bitalgo

# 2. 가상환경 생성 및 패키지 설치
pip install -r requirements.txt

# 3. 실행
streamlit run app.py

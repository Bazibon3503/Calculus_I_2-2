import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ==========================================
# 1. 기본 설정
# ==========================================

plt.rc("font", family="Malgun Gothic")
plt.rcParams["axes.unicode_minus"] = False

DEFAULT_DATE = datetime.datetime.now().strftime("%Y%m%d")
DEFAULT_TIME = "0500"
DEFAULT_NX = 60
DEFAULT_NY = 127
DEFAULT_INTERVAL = 1
DEFAULT_API_KEY = "ab24f451d29a95f1ff7642702b92557b73fb18c0841235538697e25047d16202"

API_URL = (
    "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
)


# ==========================================
# 2. 기상청 API 데이터 수집
# ==========================================
def fetch_kma_weather_data(base_date, base_time, service_key, nx, ny):
  if not service_key.strip():
    raise ValueError("API 인증키가 없습니다. 인증키를 입력하세요.")

  params = {
      "serviceKey": service_key.strip(),
      "pageNo": "1",
      "numOfRows": "1000",
      "dataType": "JSON",
      "base_date": base_date,
      "base_time": base_time,
      "nx": nx,
      "ny": ny,
  }

  try:
    response = requests.get(API_URL, params=params, timeout=20)
    response.raise_for_status()
    result = response.json()

    if "response" not in result:
      raise ValueError("API 응답 형식이 올바르지 않습니다.")

    result = result["response"]
    header = result["header"]

    if header["resultCode"] != "00":
      raise ValueError(f"기상청 API 오류: {header['resultMsg']}")

    items = result["body"]["items"]["item"]
    if isinstance(items, dict):
      items = [items]

    rows = []
    for item in items:
      if item["category"] == "TMP":
        rows.append({
            "예보일자": item["fcstDate"],
            "예보시각": item["fcstTime"],
            "기온(℃)": float(item["fcstValue"]),
        })

    data = pd.DataFrame(rows)
    if data.empty:
      raise ValueError("해당 날짜와 지역의 기온 예보가 없습니다.")

    data = data.drop_duplicates(subset=["예보일자", "예보시각"])
    data["예보일시"] = pd.to_datetime(
        data["예보일자"] + data["예보시각"], format="%Y%m%d%H%M"
    )
    data = data.sort_values("예보일시").reset_index(drop=True)
    return data

  except requests.exceptions.Timeout:
    raise RuntimeError("기상청 서버 응답 시간이 초과되었습니다.")
  except requests.exceptions.RequestException as e:
    raise RuntimeError(f"인터넷 또는 API 연결 오류: {e}")
  except (ValueError, KeyError, TypeError) as e:
    raise RuntimeError(f"기상청 데이터 처리 오류: {e}")


# ==========================================
# 3. 미분계수와 접선의 방정식
# ==========================================
def calculate_weather(data, interval):
  data = data.copy().reset_index(drop=True)
  if len(data) < 2:
    raise ValueError("기온 데이터가 2개 이상 필요합니다.")
  if interval < 1:
    raise ValueError("예측 시간은 1 이상이어야 합니다.")

  elapsed = (
      data["예보일시"] - data["예보일시"].iloc[0]
  ).dt.total_seconds().to_numpy() / 3600
  temperature = data["기온(℃)"].to_numpy()
  rate = np.gradient(temperature, elapsed)
  data["순간변화율(℃/h)"] = rate
  data["예측기온(℃)"] = np.nan

  for i in range(len(data)):
    current_time = elapsed[i]
    source_time = current_time - interval
    if source_time < elapsed[0] or source_time > elapsed[-1]:
      continue

    current_temp = np.interp(source_time, elapsed, temperature)
    current_rate = np.interp(source_time, elapsed, rate)
    predicted_temp = current_temp + current_rate * interval
    data.loc[i, "예측기온(℃)"] = predicted_temp

  return data


# ==========================================
# 4. Streamlit UI 구성
# ==========================================
st.title("🌡️ 미분 기반 단기 기온 예측 프로그램")
st.markdown("서울 기온 예측 - 미분과 접선의 방정식")

with st.sidebar:
  st.header("⚙️ 설정 입력")
  input_date = st.text_input("조회 날짜 (YYYYMMDD)", value=DEFAULT_DATE)
  input_time = st.text_input("발표 시각", value=DEFAULT_TIME, disabled=True)
  input_x = st.number_input(
      "X 좌표", min_value=1, max_value=149, value=DEFAULT_NX
  )
  input_y = st.number_input(
      "Y 좌표", min_value=1, max_value=253, value=DEFAULT_NY
  )
  input_interval = st.number_input(
      "예측 시간(h)", min_value=1, value=DEFAULT_INTERVAL
  )
  input_key = st.text_input("API 인증키", value=DEFAULT_API_KEY, type="password")

  run_button = st.button("예측 실행", type="primary")

# ==========================================
# 5. 실행 및 결과 표시
# ==========================================
if run_button:
  try:
    datetime.datetime.strptime(input_date, "%Y%m%d")

    with st.spinner("기상청 데이터를 불러오는 중입니다..."):
      df = fetch_kma_weather_data(
          input_date, input_time, input_key, int(input_x), int(input_y)
      )
      df = calculate_weather(df, int(input_interval))

    st.success("실제 예보와 수학적 모델 비교 완료!")

    # 그래프 그리기
    fig, ax = plt.subplots(figsize=(10, 5))
    x = df["예보일시"]
    ax.plot(
        x,
        df["기온(℃)"],
        "o-",
        label="기상청 실제 예보 기온",
        linewidth=2,
        markersize=5,
    )
    ax.plot(
        x,
        df["예측기온(℃)"],
        "s--",
        label=f"{input_interval}시간 후 수학적 모델 예측",
        linewidth=2,
        markersize=5,
    )
    ax.set_title(
        f"{input_date} 기온 예측: 미분과 접선의 방정식", fontsize=14
    )
    ax.set_xlabel("예보 시각")
    ax.set_ylabel("기온 (℃)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    st.pyplot(fig)

    # 데이터 테이블 출력
    st.subheader("📋 기온 예측 결과 데이터")
    st.dataframe(
        df[[
            "예보일자",
            "예보시각",
            "기온(℃)",
            "순간변화율(℃/h)",
            "예측기온(℃)",
        ]].round(2)
    )

  except Exception as e:
    st.error(f"오류가 발생했습니다: {e}")
else:
  st.info(
      "좌측 사이드바에서 조회 날짜와 API 인증키를 확인한 후 **'예측 실행'**"
      " 버튼을 누르세요."
  )
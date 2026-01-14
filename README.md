# Information-Technology-Graduation-Project

# 🇻🇳 README (Vietnamese) Developing Multimodal Models for Stock Price Forecasting
## **📖 Giới thiệu**

Dự án này là đồ án tốt nghiệp ngành Công nghệ Thông tin, tập trung xây dựng mô hình dự báo giá cổ phiếu đa phương thức bằng cách tích hợp dữ liệu chuỗi thời gian thị trường và tin tức tài chính tiếng Việt.

Khác với các phương pháp chỉ sử dụng dữ liệu giá hoặc kết hợp tin tức một cách hời hợt, dự án đề xuất các chiến lược fusion có định hướng, đặc biệt là cross-attention, nhằm giúp mô hình tự động chọn lọc các tin tức thực sự liên quan đến biến động giá.

## **Lưu ý (Note)**

Dự án hiện đang trong giai đoạn hoàn thiện và đánh giá (grading phase) của đồ án tốt nghiệp. Do đó, toàn bộ mã nguồn chưa được công khai đầy đủ tại thời điểm hiện tại.

Repository này được sử dụng nhằm trình bày ý tưởng nghiên cứu, thiết kế mô hình, quy trình thực nghiệm và kết quả chính của dự án. Mã nguồn hoàn chỉnh sẽ được cập nhật sau khi quá trình chấm điểm chính thức kết thúc.

## **🎯 Mục tiêu**

- **Phát triển các mô hình dự báo** giá cổ phiếu **đa phương thức** dựa trên: **Cross-Attention, Feature Concatenation**

- **So sánh** hiệu quả giữa: **Time-series only, Time-series + News**

- **Đánh giá** hiệu quả trên nhiều horizon dự báo: **H ∈ {1, 4, 7, 10, 14, 21}**

- **Phân tích vai trò của tin tức tài chính** trong ngắn hạn và trung hạn.

## **🗂️ Dữ liệu**
- **📈 Dữ liệu chuỗi thời gian (ACB – VN30)**

  - **Thời gian**: 01/2020 – 10/2025

  - **1,442** phiên giao dịch

  - **22 đặc trưng**, bao gồm:
    - **OHLCV**: Open, High, Low, Close, Volumn
    - **Technical indicators** (RSI, MA, MACD, ADX, STOCH, STOCHRSI)
    - **Macroeconomic indicators** (GDP, CPI, USD/VND)

## **🔒 Chống data leakage:**

  - GDP trễ 1 năm

  - CPI trễ 1 tháng

  - USD/VND trễ 1 ngày

## **📰 Dữ liệu tin tức tài chính**

  - **13,739** bài báo tài chính tiếng Việt

  - **Nguồn**: Kaggle, VnEconomy,..... và nhiều nguồn khác

  - **Lọc tin theo keyword domain-specific** (ACB, GDP, CPI, tỷ giá, thị trường…)

## **🛠️ Tiền xử lý & Feature Engineering**

**Chuẩn hóa:**

  - **RobustScaler** cho các features khác ngoài giá đóng cửa

  - **StandardScaler** cho giá đóng cửa

**So sánh 2 feature sets:**

  - **Filtered Features** (10) – tương quan Pearson ≥ 0.3

  - **Full Features** (19) – giàu tín hiệu cho trung & dài hạn

**Tách tập theo thứ tự thời gian (60/15/25)**

## **🧠 Mô hình**
  - **Time-Series Encoders:** LSTM, PatchTST, iTransformer

  - **Text Embedding Models**: Vietnamese Embedding (AITeamVN) – 1024 dim, Vietnamese Document Embedding – 768 dim

  - **Multimodal Fusion**:
    - Cross-Attention: TS hidden states ↔ News embeddings
    - Concatenation

## **⚙️ Thiết lập huấn luyện**
  - Optimizer: AdamW
  - Loss: MSE
  - Early stopping + Gradient clipping
  - Hyperparameter tuning: Optuna (TPESampler, MedianPruner)
  - Lookback window: L ∈ {12, 24, …, 96}

## **📊 Kết quả chính**

  - Filtered Features hiệu quả hơn cho H = 1, 4

  - Full Features vượt trội cho H ≥ 7

  - Multimodal (TS + News) cải thiện MAE đến ~9% ở ngắn & trung hạn

  - Cross-Attention ổn định hơn Concatenation, đặc biệt ở H = 1–7
  - **Ảnh kết quả**:
    
     <img width="814" height="513" alt="image" src="https://github.com/user-attachments/assets/b31bebfd-5378-42f6-8052-2b1298dec184" />
     <img width="833" height="468" alt="image" src="https://github.com/user-attachments/assets/e5fcd6b3-8e32-41d8-aa9a-1bcbeb5d4a4e" />
     <img width="833" height="425" alt="image" src="https://github.com/user-attachments/assets/d5552e00-00ad-4916-ae8e-ff6291f80e3c" />
     <img width="734" height="533" alt="image" src="https://github.com/user-attachments/assets/f801d270-df24-42e6-8c5f-bbb06e6eeaa5" />
     <img width="813" height="562" alt="image" src="https://github.com/user-attachments/assets/707e0936-c851-43e5-aa1d-bd14f995116b" />
     <img width="770" height="431" alt="image" src="https://github.com/user-attachments/assets/b7890c88-537c-417a-a54c-2e797749b679" />
     <img width="851" height="363" alt="image" src="https://github.com/user-attachments/assets/a6b80f54-f126-4f43-8970-d7b6831793b0" />

## **📌 Kết luận**

Dự án chứng minh rằng tích hợp tin tức tài chính một cách có chọn lọc thông qua cross-attention giúp cải thiện đáng kể độ chính xác dự báo giá cổ phiếu, đặc biệt trong ngắn hạn. Đây là một hướng tiếp cận khả thi cho các hệ thống hỗ trợ quyết định giao dịch và đầu tư.

# 🇬🇧 README (English) Developing Multimodal Models for Stock Price Forecasting
📖 Overview

This undergraduate IT graduation project proposes a multimodal stock price forecasting framework that integrates historical market time-series data with Vietnamese financial news.

Unlike conventional approaches that either rely solely on numerical data or naïvely combine text and prices, this project introduces relevance-aware multimodal fusion, particularly via cross-attention, to dynamically identify news that truly influences price movements.

🎯 Objectives

Develop multimodal forecasting models using:

Cross-Attention

Feature Concatenation

Compare time-series-only and multimodal approaches

Evaluate performance across multiple horizons:
H ∈ {1, 4, 7, 10, 14, 21}

Analyze horizon-dependent contributions of financial news.

🗂️ Datasets

Stock data: ACB (VN30), 2020–2025, OHLCV + technical + macro indicators

News data: 13,739 Vietnamese financial articles (Kaggle + VnEconomy)

Leakage prevention via lagged macroeconomic variables

🧠 Models & Methods

Time-series encoders: LSTM, PatchTST, iTransformer

Text embeddings: Vietnamese Embedding, Vietnamese Document Embedding

Fusion strategies: Cross-Attention, Concatenation

Optimization: AdamW, Optuna, Early Stopping

Evaluation: MAE, RMSE, MAPE across multiple horizons

📊 Key Findings

Filtered features perform better for short horizons

Full features dominate medium-to-long horizons

Multimodal models improve MAE by up to ~9% in short-term forecasts

Cross-attention provides more robust and selective fusion than concatenation

📌 Conclusion

The results confirm that relevance-aware news integration via cross-attention significantly enhances stock price forecasting, particularly for short- and medium-term horizons, offering practical insights for real-world financial decision-support systems.

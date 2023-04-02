# Logistic Regression
## What is the Logistic Regression?
**Logistic Regression** là một kỹ thuật dùng để dự đoán kết quả của một biến phụ thuộc dạng phân loại (ví dụ dự đoán khối u là ác tính hay lành tính,... ).

## Logistic Regression Equation
Trong hồi quy Logistic, hàm sigmoid được sử dụng nhiều nhất. Một số đặc điểm của hàm sigmoid:
- Hàm sigmoid:
  $$f(x)=\frac{1}{1+e^{-s}}≜σ(s)$$
- Hàm sigmoid bị chặn trong khoảng (0,1). Ngoài ra
$$\lim_{x\to-\infty}σ(s)=0$$ $$\lim_{x\to\infty}σ(s)=1$$
- Đặc biệt:
$$σ'(s)=σ(s)(1-σ(s))$$

## Linear vs Logistic Regression
|Linear Regression|Logistic Regression|
|---|---|
|Biến liên tục|Biến phân loại|
|Giải quyết vấn đề hồi quy|Giải quyết vấn đề phân loại|
|Đường thẳng|Đường cong S|

## Use cases
- Dự báo thời tiết
- Vấn đề phân loại
- Dự đoán bệnh

## Code in python
**Các bước:**
- **Thu thập dữ liệu:** Nhập thư viện
- **Phân tích dữ liệu:** Tạo những biểu đồ khác nhau để kiểm tra mối quan hệ của các biến.
- **Sắp xếp dự liệu:** Làm sạch bằng cách loại bỏ giá trị NA và các cột không cần thiết.
- **Huấn luyện và kiểm tra dữ liệu:** Xây dựng mô hình huấn luyện và dự đoán kết quả dựa trên dữ liệu thử nghiệm.
- **Kiểm tra độ chính xác.**


Code được viết trong file week4.ipynb

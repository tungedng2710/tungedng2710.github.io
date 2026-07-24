---
title: "Suy luận Bayes: Tiên nghiệm, hàm hợp lý và quyết định"
pubDate: 2026-01-22
image: "https://towardsdatascience.com/wp-content/uploads/2020/12/1eWB1B9MCjG7ZNxXumpsblw.png"
description: Một bài giới thiệu thực tiễn về suy luận Bayes, từ định lý Bayes, phân phối tiên nghiệm liên hợp và các phương pháp xấp xỉ đến ví dụ Beta–Binomial cụ thể.
tags:
- Thống kê Bayes
- Machine Learning
- Xác suất
authorName: Tung Nguyen
authorUrl: https://github.com/tungedng2710
lang: vi
translationKey: bayesian-inference
---

# Vì sao suy luận Bayes quan trọng

Các quy trình tất định thường hoạt động kém khi phân phối dữ liệu thay đổi hoặc lượng bằng chứng không còn như trước. Suy luận Bayes duy trì một phân phối xác suất đầy đủ cho các đại lượng chưa chắc chắn. Nhờ đó, ta có thể cập nhật niềm tin khi có quan sát mới và giữ cho các quyết định phía sau được hiệu chỉnh phù hợp.

Cách làm cơ bản khá đơn giản: mã hóa hiểu biết sẵn có bằng phân phối tiên nghiệm, mô tả cách dữ liệu được sinh ra bằng hàm hợp lý, rồi kết hợp hai thành phần qua định lý Bayes để thu được phân phối hậu nghiệm.

## Nhắc lại định lý Bayes

Ở dạng phổ biến nhất, định lý Bayes mô tả phân phối hậu nghiệm của tham số $\theta$ sau khi quan sát dữ liệu $\mathcal{D}$:

$$
 p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta)\,p(\theta)}{p(\mathcal{D})} \propto p(\mathcal{D} \mid \theta)\,p(\theta).
$$

- **Phân phối tiên nghiệm** $p(\theta)$: niềm tin trước khi quan sát dữ liệu.
- **Hàm hợp lý** $p(\mathcal{D} \mid \theta)$: mô hình sinh dữ liệu.
- **Bằng chứng** $p(\mathcal{D})$: hằng số chuẩn hóa, thường là một tích phân hoặc tổng.
- **Phân phối hậu nghiệm** $p(\theta \mid \mathcal{D})$: niềm tin đã được cập nhật, dùng cho dự đoán và ra quyết định.

Dạng tỉ lệ thường được dùng vì bằng chứng là hằng số theo $\theta$. Việc chuẩn hóa sẽ được xử lý trực tiếp hoặc thông qua các thuật toán lấy mẫu.

## Cấu trúc của một mô hình Bayes

1. **Giả định cấu trúc**: chọn họ phân phối như Bernoulli, Poisson hay Gaussian process sao cho phù hợp với quá trình đo lường.
2. **Tham số hóa**: xác định các biến ẩn và siêu tham số biểu diễn những quan hệ chưa biết.
3. **Chọn phân phối tiên nghiệm**: dùng tiên nghiệm giàu thông tin khi có kiến thức miền, hoặc tiên nghiệm ít thông tin nếu muốn giữ tính trung lập. Phân tích độ nhạy rất quan trọng: hãy thay đổi siêu tham số và kiểm tra xem hậu nghiệm có ổn định hay không.
4. **Tính hậu nghiệm**: sử dụng cập nhật giải tích, phương pháp xấp xỉ tất định hoặc bộ lấy mẫu ngẫu nhiên hoàn toàn.
5. **Phân phối dự đoán hậu nghiệm**: lấy tích phân theo $\theta$ để dự đoán và định lượng độ bất định cho dữ liệu mới.

## Phân phối tiên nghiệm liên hợp trong thực tế

Phân phối tiên nghiệm liên hợp tạo ra hậu nghiệm thuộc cùng một họ phân phối. Nhờ vậy, ta có công thức cập nhật đóng với chi phí tính toán rất thấp:

| Hàm hợp lý | Tiên nghiệm liên hợp | Tham số hậu nghiệm |
| --- | --- | --- |
| Bernoulli/Binomial với xác suất $p$ | $\text{Beta}(\alpha, \beta)$ | $\text{Beta}(\alpha + k, \beta + n - k)$ |
| Poisson với tốc độ $\lambda$ | $\text{Gamma}(a, b)$ | $\text{Gamma}(a + \sum x_i, b + n)$ |
| Gaussian với phương sai đã biết | Tiên nghiệm Gaussian | Cập nhật trung bình và phương sai bằng cách cộng độ chính xác |

Tính liên hợp đặc biệt phù hợp với dashboard, phân tích dữ liệu luồng hoặc hệ thống nhúng, nơi cần cập nhật trong vài micro giây mà không phải chạy cả một quy trình suy luận.

## Ví dụ: cập nhật Beta–Binomial

Giả sử bạn đang theo dõi tỉ lệ nhấp chuột. Ta bắt đầu bằng một tiên nghiệm Beta trung lập:

- Tiên nghiệm: $p(p) = \text{Beta}(1, 1)$, tức phân phối đều từ 0 đến 1.
- Quan sát: $n = 40$ lượt hiển thị và $k = 14$ lượt nhấp.

Các tham số hậu nghiệm là $\alpha' = 1 + 14 = 15$ và $\beta' = 1 + 40 - 14 = 27$, do đó:

$$
 p(p \mid \mathcal{D}) = \text{Beta}(15, 27).
$$

Từ phân phối hậu nghiệm này, ta có thể tính:

- Trung bình hậu nghiệm: $15/(15 + 27) \approx 0.357$. Giá trị này thấp hơn một chút so với tỉ lệ thực nghiệm vì tiên nghiệm đã bổ sung các số đếm giả.
- Khoảng tin cậy Bayes 95%: lấy các phân vị 2,5% và 97,5% của phân phối $\text{Beta}(15, 27)$, xấp xỉ $[0.23, 0.49]$.
- Dự đoán hậu nghiệm cho lượt hiển thị tiếp theo: $p(\text{nhấp}) = \frac{15}{15 + 27} \approx 0.357$.

## Vượt ra ngoài công thức đóng: suy luận Bayes xấp xỉ

Phần lớn mô hình thực tế như hồi quy phân cấp, mạng nơ-ron Bayes hay mô hình không gian trạng thái đều không có tính liên hợp. Một số chiến lược xấp xỉ phổ biến gồm:

- **Xấp xỉ Laplace**: khớp một phân phối Gaussian quanh ước lượng hậu nghiệm cực đại (MAP), sử dụng ma trận Hessian của log-hậu nghiệm.
- **Suy luận biến phân (VI)**: giả sử một họ phân phối dễ xử lý $q_\phi(\theta)$ rồi cực tiểu hóa phân kỳ KL $\mathrm{KL}(q_\phi \Vert p)$. VI ngẫu nhiên với các mini-batch có thể mở rộng tới hàng triệu điểm dữ liệu.
- **Markov Chain Monte Carlo (MCMC)**: lấy các mẫu có phân phối dừng bằng phân phối hậu nghiệm. Hamiltonian Monte Carlo và biến thể động NUTS vẫn là những lựa chọn mặc định mạnh cho mô hình cỡ vừa.
- **Sequential Monte Carlo / bộ lọc hạt**: cập nhật trực tuyến một tập mẫu có trọng số, hữu ích cho bài toán theo dõi và dữ liệu luồng.

Trong hệ thống thực tế, các phương pháp này thường được kết hợp với nhau: dùng suy luận biến phân để khởi tạo, sau đó chạy các chuỗi MCMC ngắn để nắm bắt tốt hơn hành vi ở phần đuôi phân phối.

## Danh sách kiểm tra cho quy trình Bayes

1. **Phản biện mô hình**: mô phỏng dữ liệu tổng hợp từ phân phối dự đoán tiên nghiệm. Dữ liệu đó có giống các quan sát hợp lý hay không?
2. **Chẩn đoán suy luận**: theo dõi cỡ mẫu hiệu dụng, $\hat{R}$, chuẩn gradient hoặc độ hội tụ ELBO tùy theo phương pháp.
3. **Kiểm tra dự đoán hậu nghiệm**: lấy mẫu $y^{(rep)}$ rồi so sánh các thống kê tóm tắt hoặc thước đo sai khác với dữ liệu thật.
4. **Phân tích quyết định**: lấy tích phân của hàm lợi ích hoặc mất mát theo hậu nghiệm. Quyết định sẽ thay đổi khi hậu nghiệm vượt qua các ngưỡng rủi ro đã định trước.
5. **Truyền đạt kết quả**: tóm tắt phân phối hậu nghiệm bằng trung vị và khoảng tin cậy Bayes thay vì chỉ dùng MAP, đồng thời trực quan hóa ảnh hưởng của tiên nghiệm đến kết quả.

## Khi nào nên dùng suy luận Bayes?

Phương pháp Bayes đặc biệt hữu ích khi độ bất định đóng vai trò quan trọng:

- **Dữ liệu khan hiếm**: tiên nghiệm giúp ổn định ước lượng khi chỉ có ít quan sát.
- **Cấu trúc phân cấp**: gom thông tin giữa các nhóm liên quan giúp cải thiện ước lượng cho những phân khúc có tín hiệu yếu.
- **Ra quyết định tuần tự**: cập nhật niềm tin theo thời gian thực rất phù hợp với thí nghiệm thích ứng, robot và phát hiện bất thường.
- **Môi trường có quy định chặt chẽ**: khoảng tin cậy Bayes và tiên nghiệm được khai báo rõ ràng giúp quá trình kiểm toán minh bạch hơn.

Nếu chi phí tính toán là mối lo, hãy bắt đầu với mô hình liên hợp hoặc suy luận biến phân. Sau đó, khi ứng dụng đã chứng minh được giá trị, bạn có thể mở rộng sang các bộ lấy mẫu giàu biểu diễn hơn.

## Đọc thêm

- *Bayesian Data Analysis* của Gelman và cộng sự, dành cho quy trình Bayes và kiểm tra mô hình.
- *Probabilistic Machine Learning* của Murphy, dành cho các công cụ VI và MCMC hiện đại.
- Cộng đồng Stan, NumPyro, PyMC và Bean Machine, nơi có nhiều ví dụ và hướng dẫn chẩn đoán chất lượng cao.

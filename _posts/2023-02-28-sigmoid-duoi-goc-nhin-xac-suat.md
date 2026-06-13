---
title: Hàm sigmoid dưới góc nhìn xác suất
layout: post
post-image: "/assets/images/posts/sigmoid-curve.svg"
description: Giải thích hàm sigmoid qua log-odds, phân phối Bernoulli, maximum likelihood và định lý Bayes.
tags:
- Sigmoid
- Probability and Statistics
author-name: Tung Nguyen
author-url: https://github.com/tungedng2710
---

Trong bài toán phân loại nhị phân, mô hình thường tạo ra một điểm số thực
$z \in \mathbb{R}$. Tuy nhiên, một số thực bất kỳ chưa thể được diễn giải trực
tiếp như xác suất. Hàm sigmoid giải quyết vấn đề này bằng cách ánh xạ $z$ vào
khoảng $(0, 1)$:

\[
\sigma(z) = \frac{1}{1 + e^{-z}}.
\]

Nếu đặt

\[
P(y=1 \mid x) = \sigma(z),
\]

thì xác suất của lớp còn lại là

\[
P(y=0 \mid x) = 1 - \sigma(z).
\]

Khác với softmax, sigmoid trong phân loại nhị phân chỉ cần trả về **một giá trị
vô hướng**. Hai xác suất có tổng bằng $1$ là $\sigma(z)$ và $1-\sigma(z)$, chứ
không phải các phần tử trong đầu ra của riêng hàm sigmoid.

Bài viết này sẽ giải thích vì sao sigmoid xuất hiện tự nhiên trong mô hình xác
suất, thay vì chỉ xem nó như một công thức dùng để ép đầu ra vào khoảng từ $0$
đến $1$.

## 1. Từ odds đến hàm sigmoid

Gọi

\[
p(x) = P(y=1 \mid x).
\]

**Odds** của lớp $1$ so với lớp $0$ được định nghĩa là

\[
\frac{p(x)}{1-p(x)}.
\]

Odds thuộc khoảng $(0, +\infty)$ nên vẫn chưa thuận tiện để mô hình hóa bằng
một hàm tuyến tính. Lấy logarithm, ta thu được **log-odds**, còn gọi là
**logit**:

\[
\operatorname{logit}(p(x))
= \log\frac{p(x)}{1-p(x)}.
\]

Logit nhận giá trị trên toàn bộ trục số thực. Logistic regression giả sử
log-odds là một hàm tuyến tính của vector đặc trưng:

\[
\log\frac{p(x)}{1-p(x)} = w^Tx+b.
\]

Đặt $z=w^Tx+b$ rồi giải phương trình theo $p(x)$:

\[
\frac{p(x)}{1-p(x)}=e^z,
\]

\[
p(x)=e^z(1-p(x)),
\]

\[
p(x)(1+e^z)=e^z,
\]

\[
p(x)=\frac{e^z}{1+e^z}
=\frac{1}{1+e^{-z}}
=\sigma(z).
\]

Như vậy, sigmoid chính là **hàm ngược của logit**. Nó xuất hiện khi ta giả sử
log-odds của xác suất thuộc lớp $1$ thay đổi tuyến tính theo đặc trưng đầu vào.

Một vài tính chất quan trọng:

- $\sigma(z) \in (0,1)$ với mọi $z \in \mathbb{R}$;
- $\sigma(0)=0.5$;
- $\sigma(-z)=1-\sigma(z)$;
- $\sigma'(z)=\sigma(z)(1-\sigma(z))$.

## 2. Maximum likelihood và binary cross-entropy

Xét tập dữ liệu

\[
\mathcal{D}=\{(x^{(i)},y^{(i)})\}_{i=1}^n,
\qquad y^{(i)}\in\{0,1\}.
\]

Với mỗi quan sát, đặt

\[
p_i=P(y^{(i)}=1\mid x^{(i)};w,b)
=\sigma(w^Tx^{(i)}+b).
\]

Vì $y^{(i)}$ là biến nhị phân, ta mô hình hóa nó bằng phân phối Bernoulli:

\[
P(y^{(i)}\mid x^{(i)};w,b)
=p_i^{y^{(i)}}(1-p_i)^{1-y^{(i)}}.
\]

Giả sử các quan sát độc lập có điều kiện khi biết đầu vào, likelihood của toàn
bộ tập dữ liệu là

\[
L(w,b)
=\prod_{i=1}^n
p_i^{y^{(i)}}(1-p_i)^{1-y^{(i)}}.
\]

Trong maximum likelihood estimation (MLE), $w$ và $b$ là các tham số cố định
nhưng chưa biết. Ta chọn giá trị của chúng sao cho dữ liệu quan sát có
likelihood lớn nhất. Lấy logarithm:

\[
\ell(w,b)
=\log L(w,b)
=\sum_{i=1}^n
\left[
y^{(i)}\log p_i
+(1-y^{(i)})\log(1-p_i)
\right].
\]

Cực đại hóa log-likelihood tương đương với cực tiểu hóa negative
log-likelihood:

\[
\mathcal{L}(w,b)
=-\sum_{i=1}^n
\left[
y^{(i)}\log p_i
+(1-y^{(i)})\log(1-p_i)
\right].
\]

Đây chính là **binary cross-entropy loss** thường dùng để huấn luyện mô hình
phân loại nhị phân. Vì vậy, sigmoid và binary cross-entropy không phải hai lựa
chọn rời rạc: chúng tạo thành mô hình xác suất Bernoulli được ước lượng bằng
MLE.

## 3. Suy ra sigmoid từ định lý Bayes

Ta cũng có thể thấy sigmoid xuất hiện trong một mô hình sinh
(*generative model*).

Xét hai lớp $C_0$ và $C_1$ với xác suất tiên nghiệm

\[
P(C_0)=\pi_0,\qquad P(C_1)=\pi_1,
\qquad \pi_0+\pi_1=1.
\]

Giả sử phân phối của dữ liệu trong mỗi lớp là Gaussian và hai lớp dùng chung
ma trận hiệp phương sai $\Sigma$:

\[
x\mid C_0 \sim \mathcal{N}(\mu_0,\Sigma),
\]

\[
x\mid C_1 \sim \mathcal{N}(\mu_1,\Sigma).
\]

Theo định lý Bayes, posterior odds là

\[
\frac{P(C_1\mid x)}{P(C_0\mid x)}
=\frac{p(x\mid C_1)\pi_1}{p(x\mid C_0)\pi_0}.
\]

Lấy logarithm hai vế:

\[
\log\frac{P(C_1\mid x)}{P(C_0\mid x)}
=
\log\frac{p(x\mid C_1)\pi_1}
{p(x\mid C_0)\pi_0}.
\]

Thay mật độ Gaussian vào biểu thức trên. Vì hai lớp có cùng $\Sigma$, các
thành phần bậc hai theo $x$ triệt tiêu, để lại

\[
\log\frac{P(C_1\mid x)}{P(C_0\mid x)}
=w^Tx+b,
\]

trong đó

\[
w=\Sigma^{-1}(\mu_1-\mu_0)
\]

và

\[
b=
-\frac{1}{2}\mu_1^T\Sigma^{-1}\mu_1
+\frac{1}{2}\mu_0^T\Sigma^{-1}\mu_0
+\log\frac{\pi_1}{\pi_0}.
\]

Đặt $p(x)=P(C_1\mid x)$. Vì $P(C_0\mid x)=1-p(x)$, ta có

\[
\log\frac{p(x)}{1-p(x)}=w^Tx+b.
\]

Theo kết quả ở phần 1:

\[
P(C_1\mid x)=\sigma(w^Tx+b).
\]

Đây là mối liên hệ giữa Gaussian discriminant analysis và logistic
regression: nếu hai mật độ Gaussian theo lớp có chung ma trận hiệp phương sai,
posterior có dạng sigmoid của một hàm tuyến tính.

Điều kiện dùng chung $\Sigma$ rất quan trọng. Nếu mỗi lớp có một ma trận hiệp
phương sai khác nhau, log-odds nói chung là một hàm bậc hai theo $x$, và biên
quyết định không còn tuyến tính.

## 4. Mở rộng sang nhiều lớp: softmax

Với $K$ lớp loại trừ lẫn nhau, mô hình tạo ra $K$ điểm số
$z_1,\dots,z_K$. Softmax chuyển chúng thành một phân phối xác suất:

\[
P(y=k\mid x)
=\frac{e^{z_k}}
{\sum_{j=1}^{K}e^{z_j}}.
\]

Các đầu ra của softmax đều thuộc $(0,1)$ và có tổng bằng $1$. Softmax có thể
được xem là sự mở rộng của logistic regression sang nhiều lớp, nhưng không nên
gọi đơn giản là “sigmoid cho nhiều lớp”.

Trong trường hợp hai lớp:

\[
P(y=1\mid x)
=\frac{e^{z_1}}{e^{z_0}+e^{z_1}}
=\frac{1}{1+e^{-(z_1-z_0)}}
=\sigma(z_1-z_0).
\]

Do đó, softmax hai lớp tương đương với sigmoid áp dụng lên hiệu của hai
logit.

## 5. Một lưu ý về cách hiểu xác suất

Sigmoid bảo đảm đầu ra nằm trong khoảng $(0,1)$ và cho phép xây dựng một mô
hình xác suất nhất quán. Tuy nhiên, điều đó không bảo đảm xác suất dự đoán luôn
được **hiệu chỉnh tốt** (*well-calibrated*).

Ví dụ, nếu mô hình dự đoán $0.8$ cho nhiều quan sát tương tự, một mô hình được
hiệu chỉnh tốt cần có khoảng $80\%$ số quan sát đó thực sự thuộc lớp $1$. Chất
lượng hiệu chỉnh còn phụ thuộc vào dữ liệu, giả định mô hình, regularization và
quy trình huấn luyện.

## Kết luận

Hàm sigmoid xuất hiện tự nhiên trong phân loại nhị phân vì ba lý do liên hệ
chặt chẽ với nhau:

1. Sigmoid là hàm ngược của logit, nên biến một log-odds bất kỳ thành xác suất.
2. Kết hợp sigmoid với phân phối Bernoulli và MLE dẫn đến binary
   cross-entropy.
3. Theo Bayes, hai phân phối Gaussian theo lớp có chung ma trận hiệp phương sai
   tạo ra posterior là sigmoid của một hàm tuyến tính.

Nói ngắn gọn, logistic regression mô hình hóa tuyến tính trên **log-odds**, chứ
không mô hình hóa tuyến tính trực tiếp trên xác suất.

## Tài liệu tham khảo

1. C. M. Bishop. *Pattern Recognition and Machine Learning*. Springer, 2006.
2. Stanford CS229. [Machine Learning Lecture Notes](https://cs229.stanford.edu/main_notes.pdf).
3. scikit-learn. [Log loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html).
4. Tống Đình Quỳ. *Giáo trình Xác suất thống kê*. NXB Bách Khoa Hà Nội.

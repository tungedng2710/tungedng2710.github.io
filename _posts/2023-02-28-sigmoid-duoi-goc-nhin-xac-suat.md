---
title: Sigmoid dưới góc nhìn xác suất
layout: post
post-image: "https://images.viblo.asia/100x100/38326b12-21d0-4a55-b299-70c29eca1c2c.png"
description: Giải thích công thức toán học của Sigmoid trong Machine Learning
tags:
- sample
- post
- test
---

Trong các bài toán phân lớp (classification) sử dụng Deep Learning, phần không thể thiếu cho đầu ra chính là hàm sigmoid (cho 2 classes) hay softmax (biến thể của sigmoid cho nhiều classes). Mục đích của hàm sigmoid là cho ra output là một vector, trong đó tổng các phần tử bằng 1, biểu thị xác suất khả năng mà sample đang xét rơi vào từng class. Vậy tại sao sigmoid lại có thể biểu diễn xác suất, chúng ta cùng làm rõ trong bài viết này 😄.

## 1. Xác suất có điều kiện và Bayes 
Xác suất có điều kiện (Conditional probability) là xác suất của một sự kiện A nào đó, biết rằng một sự kiện B đã xảy ra và được ký hiệu là $P(A|B)$. Trong Deep Learning, cụ thể là bài toán classification, ta cần phải đi tìm xác suất rơi vào một class nào đó với điều kiện đã biết là vector đặc trưng của đầu vào. Mô tả rõ hơn thì với bài toán cần phân loại n classes $C_1, C_2,\dots, C_n$, mô hình nhận đầu vào (ảnh, text, âm thanh), các layers trong network sẽ trích xuất các đặc trưng của đầu vào và cho ra một vector embedding $x$. Vector embedding này sẽ điều kiện để tính xác suất rơi vào class nào. Với $c_i$ là class thứ $i$, ta có thể ký hiệu mô hình xác suất này là $P(C_i|x)$.

Quay lại về phần lý thuyết xác suất, ta cần nhắc lại công thức Bayes, đây là một lý thuyết quan trọng trong machine learning. Trước hết ta cần định nghĩa một nhóm đầy đủ như sau: Nhóm các sự kiện $A_1, A_2,\dots,A_n$ trong đó $n\geq2$ tạo thành một nhóm đầy đủ nếu
* $A_i$ và $A_j$ xung khắc từng đôi $\forall i\neq j$ hay ký hiệu là $A_iA_j=V$ với $V$ là sự kiện bất khả
* $A_1 +  A_2 + \dots + A_n = U$ (Với $U$ là sự kiện tất yếu)

Ví dụ về một nhóm đầy đủ là gieo một con xúc sắc, thì 6 sự kiện ứng với mỗi sự xuất hiện của 1 mặt trên tổng số 6 mặt tạo thành một nhóm đầy đủ. Và dễ thấy tổng xác suất mỗi sự kiện xảy ra độc lập trong một nhóm đầy đủ bằng 1. Đối chiếu với bài toán classification, với $C_i$ là sự kiện mô hình dự đoán đầu vào là class $i$, ta có $C_1, C_2,\dots, C_n$ tạo thành một nhóm đầy đủ. Với nhóm đầy đủ và H là một sự kiện nào đó, ta có công thức xác suất đầy đủ như sau:
$$P(H) = \sum_{i=i}^nP(A_i)P(H|A_i)$$

Từ công thức trên và công thức nhân xác suất ta có công thức Bayes:
$$P(A_i|H) = \frac{P(A_i)P(H|A_i)}{\sum_{i=1}^n P(A_i)P(H|A_i)}$$

## 2. Ước lượng hợp lý cực đại
Bài toán ước lượng tham số có thể phát biểu như sau: Cho biến ngẫu nhiên $X$ có luật phân phối xác suất đã biết nhưng chưa biết tham số $\theta$ nào đó, ta phải xác định giá trị của $\theta$ dựa trên các thông tin thu được từ mẫu quan sát $x_1, x_2,\dots, x_n$ của $X$. Quá trình đi xác định một tham số $\theta$ chưa biết được gọi là quá trình ước lượng tham số .
Ước lượng hợp lý cực đại (Tiếng Anh: Maximum likelihood estimation, viết tắt: MLE) là một phương pháp ước tính các tham số của phân phối xác suất giả định, dựa trên một số dữ liệu quan sát. Điều này đạt được bằng cách tối đa hóa hàm hợp lý (likelihood) sao cho, theo mô hình thống kê giả định, dữ liệu quan sát là có xác suất xảy ra lớn nhất. Nguyên lý hợp lý nhất là tìm giá trị của $\theta$ là hàm của quan sát $(x_1,\dots,x_n)$ sao cho bảo đảm xác suất thu được quan sát đó là lớn nhất. Giả sử biến gốc $X$ có hàm mật độ $f(x, \theta)$, khi đó hàm hợp lý (likelihood) được định nghĩa như sau:
$$L(x, \theta) = \prod_{i=1}^nf(x, \theta)$$

Nếu hàm likelihood đảm bảo điều kiện khả vi 2 lần , ta có điều kiện cần để có cực trị:
$$\frac{\partial L(x, \theta)}{\partial\theta}=0$$
tương đương với
$$\frac{\partial\ln L(x, \theta)}{\partial\theta}=0$$
Phương trình trên cũng được gọi là phương trình hợp lý nhất. 

*Note: Thực tế việc tìm ước lượng hợp lý nhất rất khó khăn do hàm likelihood không phải lúc nào cũng lồi và thường phi tuyến. Trong phạm vi bài viết mình sẽ không đề cập quá sâu phần này, các bạn có thể tìm hiểu thêm*

## 3. Mô hình hồi quy tuyến tính dưới góc nhìn xác suất
Lý do đề cập đến hồi quy tuyến tính ở đây là để xem cách chúng ta có thể xem nó như một mô hình xác suất của dữ liệu và liệu chúng ta có thể áp dụng các ý tưởng tương tự vào việc phân loại hay không. Giả sử ta có mô hình tuyến tính sau
$$y^{(i)}=\theta^Tx^{(i)}+\epsilon^{(i)}$$
Trong đó $\epsilon$ là nhiễu và độc lập với $x$. Theo Định lý giới hạn trung tâm (Định lý này là kết quả về sự hội tụ yếu của một dãy các biến ngẫu nhiên, theo đó tổng của các biến ngẫu nhiên độc lập và phân phối đồng nhất theo cùng một phân phối xác suất, sẽ hội tụ về một biến ngẫu nhiên nào đó) thì $\epsilon$ tuân theo phân phối chuẩn. Ở đây $\epsilon$ được dùng để biểu thị sự sai lệch giữa target và kết quả dự đoán của mô hình, và nó có hàm phân phối
$$P(\epsilon^{(i)})=\frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(\epsilon^{(i)})^2}{2\sigma^2})$$
$$P(y^{(i)}|x^{(i)}; \theta)=\frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2})$$
Ta gọi đây là xác suất xảy ra $y^{(i)}$ với điều kiện $x^{(i)}$ được tham số hóa bởi $\theta$. Ở đây bộ trọng số $\theta$ là biến ngẫu nhiên và được tối ưu trong quá trình mô hình học. Một cách tự nhiên, mục đích của quá trình học là tìm bộ tham số $\theta$ sao cho với đặc trưng quan sát $x^{(i)}$, khả năng nó rơi vào target $y^{(i)}$ là lớn nhất (Đến đây chúng ta đã thấy sự liên quan tới lý thuyết MLE ở bên trên rồi ha 😅). Tiếp theo ta định nghĩa hàm likelihood của $\theta$  
$$L(\theta)=L(\theta; X, \hat{y}) = P(\hat{y}|X; \theta)$$
Theo phương trình hợp lý nhất ở phần trên, ta có
$$L(\theta)=\prod_{i=1}^nP(y^{(i)}|x^{(i)}; \theta)$$
$$L(\theta)=\prod_{i=1}^n\frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2})$$
Để dễ dàng hơn cho việc tối ưu, ta lấy logarithm hai vế (việc này okela vì hàm log là đơn điệu) để biến việc tối ưu hàm mũ thành tối ưu hàm đa thức. Do đó quá trình tối ưu MLE sử dụng hàm có tên gọi là log-likelihood.
$$l(\theta) = \log L(\theta)$$
$$= \log\prod_{i=1}^n\frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2})$$
$$= \prod_{i=1}^n\log\frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2})$$
$$= n\log\frac{1}{\sqrt{2\pi}\sigma}-\frac{1}{\sigma^2}\cdot\frac{1}{2}\sum_{i=1}^{n}(y^{(i)}-\theta^Tx^{(i)})^2$$
Cực đại hóa biểu thức trên tương đương với cực tiểu hóa biểu thức sau (để ý đây chính là công thức least-square)
$$\frac{1}{2}\sum_{i=1}^{n}(y^{(i)}-\theta^Tx^{(i)})^2$$
## 4. Xây dựng công thức cho sigmoid
Ở phần 3. đã map một bộ dự đoán tuyến tính với nhiễu Gauss tới biến mục tiêu. Đối với bài toán binary classification, sẽ thật tuyệt nếu chúng ta có thể làm điều gì đó tương tự, tức là map một bộ dự đoán tuyến tính với một thứ gì đó tới xác suất thuộc một trong hai lớp và sử dụng MLE để giải thích cho thiết kế mô hình bằng việc nó tối đa hóa xác suất cho việc rơi vào 1 class nào đó của 1 đặc trưng quan sát.

Xét mô hình tuyến tính $y=\theta^Tx^{(i)}+\epsilon^{(i)}$, một tập các điểm dữ liệu thuộc 2 classes $C_1$ và $C_2$ tuân theo hai phân phối chuẩn có kỳ vọng $\mu_1$ và $\mu_2$, độ lệch chuẩn cùng bằng 1 (Ở đây mình chọn 1 cho thuận tiện việc biến đổi biểu thức, vì thực ra chọn $\sigma$ bất kỳ thì kết quả cuối cùng vẫn ra một đa thức nhưng nhìn nó sẽ rối mắt không cần thiết). Ta cần phân loại xem với với vector đặc trưng $x$, nó sẽ rơi vào class nào trong 2 classes $C_1$ và $C_2$. Theo công thức Bayes, ta có
$$P(C_1|x) = \frac{P(x|C_1)P(C_1)}{P(x|C_1)P(C_1)+P(x|C_2)P(C_2)}$$
Chia cả tử và mẫu cho $P(x|C_1)P(C_1)$ ta có: 
$$P(C_1|x) = \frac{1}{1+\frac{P(x|C_2)P(C_2)}{P(x|C_1)P(C_1)}}$$
MÌnh sẽ để lại nó một chút, quay lại về các điểm dữ liệu tuân theo hai phân phối chuẩn vừa đề cập bên trên, kết hợp sử dụng ước lượng hợp lý cực đại, ta có
$$P(x|C_1) = N(\mu_1|1) \sim \exp(-\frac{(x-\mu_1)^2}{2})$$
$$P(x|C_2) = N(\mu_2|1) \sim \exp(-\frac{(x-\mu_2)^2}{2})$$
Chia $P(X|C_2)$ cho $P(X|C_1)$ ta được:
$$ \frac{P(X|C_2)}{P(X|C_1)}=\exp(\frac{(x-\mu_1)^2-(x-\mu_2)^2}{2})$$
$$=\exp((\mu_2-\mu_1)x-\frac{1}{2}(\mu_2^2-\mu_1^2))$$
Đặt $(\mu_2-\mu_1)x-\frac{1}{2}(\mu_2^2-\mu_1^2)=-\alpha(x)$, ta thu được
$$P(C_1|x) = \frac{1}{1+\exp(-\alpha(x))} = \sigma\circ\alpha(x) $$
Trong đó $\sigma: R\to (0,1)$ được gọi là hàm sigmoid và $\alpha: R^d\to R$ được cho bởi công thức
$$\alpha(x)=\log\frac{P(x|C_1)P(C_1)}{P(x|C_2)P(C_2)}=\log\frac{P(C_1, x)}{P(C_2, x)}$$
Vậy là công thức hàm sigmoid đã xuất hiện =))). Trong trường hợp tổng quát cho nhiều class $C_1,\dots, C_n$, ta có
$$P(C_k|x)=\frac{P(x|C_k)P(C_k)}{\sum_{j=1}^{n}P(x|C_j)P(C_j)}=\frac{\exp(\alpha_k(x))}{\sum_{j=1}^{n}\exp(\alpha_j(x))}$$
Đây chính là công thức của hàm softmax!
## Tài liệu tham khảo
1. Tống Đình Quỳ. *Giáo trình xác suất thống kê*. NXB Bách Khoa Hà Nội
2. C. M. Bishop. *Pattern recognition and machine learning*. Springer, 2006.

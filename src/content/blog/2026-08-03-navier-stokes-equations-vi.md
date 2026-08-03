---
title: "Phương trình Navier–Stokes: Từ định luật bảo toàn đến dòng chảy"
pubDate: 2026-08-03
image: "/assets/images/posts/navier-stokes-flow.svg"
description: Bài nhập môn thực tiễn về hệ Navier–Stokes không nén được, ý nghĩa vật lý, nghiệm dòng chảy trong kênh, dạng không thứ nguyên và phương pháp chiếu để giải số.
tags:
- Cơ học chất lưu
- Phương trình đạo hàm riêng
- Động lực học chất lưu tính toán
- Toán ứng dụng
authorName: Tung Nguyen
authorUrl: https://github.com/tungedng2710
lang: vi
translationKey: navier-stokes-equations
---

# Vì sao hệ phương trình này quan trọng?

Phương trình Navier–Stokes mô tả sự biến đổi của vận tốc, áp suất, mật độ và nhiệt độ trong chất lưu. Chúng là nền tảng của dự báo thời tiết, thiết kế máy bay, mô phỏng dòng máu, tuần hoàn đại dương, quá trình cháy và dòng nước trong đường ống. Các thành phần xuất phát từ những nguyên lý quen thuộc—định luật II Newton, bảo toàn khối lượng và mô hình ứng suất nhớt—nhưng tương tác phi tuyến giữa chúng có thể tạo ra xoáy, lớp biên và nhiễu loạn trên rất nhiều thang không gian.

Bài viết tập trung vào chất lưu không nén được với các tính chất không đổi. Mô hình này phù hợp khi biến thiên mật độ không đáng kể, chẳng hạn nhiều dòng nước và dòng khí tốc độ thấp. Với dòng nén được, ta còn phải ghép phương trình động lượng với bảo toàn khối lượng, năng lượng và phương trình trạng thái.

## Các trường và giả định

Tại mỗi vị trí $\mathbf{x}$ và thời điểm $t$, ta cần tìm:

- trường vận tốc $\mathbf{u}(\mathbf{x},t)$, đơn vị $\mathrm{m\,s^{-1}}$;
- trường áp suất $p(\mathbf{x},t)$, đơn vị $\mathrm{Pa}$;
- mật độ $\rho$, đơn vị $\mathrm{kg\,m^{-3}}$;
- độ nhớt động lực $\mu$, đơn vị $\mathrm{Pa\,s}$.

Độ nhớt động học là $\nu=\mu/\rho$, có đơn vị $\mathrm{m^2\,s^{-1}}$. Ta xem chất lưu là một môi trường liên tục: mỗi điểm tính toán biểu diễn trung bình của rất nhiều phân tử, nhờ đó các trường khả vi có ý nghĩa.

Đạo hàm vật chất đi theo một phần tử chất lưu đang chuyển động:

$$
\frac{D}{Dt}=\frac{\partial}{\partial t}+\mathbf{u}\cdot\nabla.
$$

Nó kết hợp biến thiên cục bộ tại một điểm cố định và biến thiên do phần tử chất lưu di chuyển qua gradient không gian.

## Bảo toàn khối lượng

Với chất lưu nén được, bảo toàn khối lượng cục bộ cho phương trình liên tục:

$$
\frac{\partial \rho}{\partial t}+\nabla\cdot(\rho\mathbf{u})=0.
$$

Nếu mật độ không đổi, phương trình rút gọn thành ràng buộc không nén được:

$$
\boxed{\nabla\cdot\mathbf{u}=0.}
$$

Điều này không có nghĩa vận tốc là hằng số. Nó nói rằng một thể tích vật chất nhỏ không giãn ra hay co lại. Lượng chất lưu đi vào một thể tích kiểm soát phải rời khỏi đó với cùng tốc độ.

## Bảo toàn động lượng

Định luật II Newton cho môi trường liên tục phát biểu rằng khối lượng nhân gia tốc bằng tổng lực mặt và lực khối:

$$
\rho\frac{D\mathbf{u}}{Dt}=\nabla\cdot\boldsymbol{\sigma}+\rho\mathbf{f},
$$

trong đó $\boldsymbol{\sigma}$ là tensor ứng suất Cauchy và $\mathbf{f}$ là lực khối trên một đơn vị khối lượng, chẳng hạn trọng lực. Tách ứng suất thành phần áp suất và thành phần nhớt:

$$
\boldsymbol{\sigma}=-p\mathbf{I}+\boldsymbol{\tau}.
$$

Với chất lưu Newton:

$$
\boldsymbol{\tau}
=\mu\left(\nabla\mathbf{u}+(\nabla\mathbf{u})^T\right)
+\lambda(\nabla\cdot\mathbf{u})\mathbf{I}.
$$

Khi $\rho$ và $\mu$ không đổi, đồng thời $\nabla\cdot\mathbf{u}=0$, ta thu được

$$
\boxed{
\frac{\partial\mathbf{u}}{\partial t}
+(\mathbf{u}\cdot\nabla)\mathbf{u}
=-\frac{1}{\rho}\nabla p
+\nu\nabla^2\mathbf{u}
+\mathbf{f},
\qquad
\nabla\cdot\mathbf{u}=0.
}
$$

### Đọc từng hạng tử

| Hạng tử | Ý nghĩa |
| --- | --- |
| $\partial\mathbf{u}/\partial t$ | Gia tốc cục bộ tại một điểm cố định |
| $(\mathbf{u}\cdot\nabla)\mathbf{u}$ | Gia tốc đối lưu; vận tốc tự vận chuyển chính nó |
| $-\nabla p/\rho$ | Gia tốc do chênh lệch áp suất |
| $\nu\nabla^2\mathbf{u}$ | Khuếch tán động lượng do độ nhớt |
| $\mathbf{f}$ | Lực khối trên một đơn vị khối lượng |

Hạng tử đối lưu là phi tuyến vì vận tốc chưa biết nhân với gradient của chính nó. Áp suất có vai trò đặc biệt trong dòng không nén được: nó tự điều chỉnh để vận tốc sau cập nhật vẫn không có phân kỳ. Về mặt toán học, áp suất hoạt động như một nhân tử Lagrange áp đặt $\nabla\cdot\mathbf{u}=0$.

## Dạng không thứ nguyên và số Reynolds

Chọn chiều dài đặc trưng $L$ và vận tốc đặc trưng $U$, rồi đặt

$$
\mathbf{x}=L\mathbf{x}^*,\qquad
t=\frac{L}{U}t^*,\qquad
\mathbf{u}=U\mathbf{u}^*,\qquad
p=\rho U^2p^*.
$$

Sau khi thế vào phương trình và bỏ dấu sao, phương trình không có ngoại lực trở thành

$$
\frac{\partial\mathbf{u}}{\partial t}
+(\mathbf{u}\cdot\nabla)\mathbf{u}
=-\nabla p+\frac{1}{\mathrm{Re}}\nabla^2\mathbf{u},
$$

với

$$
\boxed{\mathrm{Re}=\frac{\rho UL}{\mu}=\frac{UL}{\nu}.}
$$

Số Reynolds so sánh vận chuyển quán tính với khuếch tán nhớt. Dòng có $\mathrm{Re}$ thấp thường trơn và bị tiêu tán mạnh; dòng có $\mathrm{Re}$ cao có thể xuất hiện lớp cắt mỏng, bất ổn và nhiễu loạn. Số Reynolds không tự quyết định toàn bộ dòng chảy—hình học, ngoại lực và điều kiện biên cũng rất quan trọng—nhưng đây là tham số tương tự đầu tiên cần kiểm tra.

## Điều kiện đầu và điều kiện biên

Một phương trình vi phân chưa tạo thành bài toán dòng chảy hoàn chỉnh nếu chưa có miền tính và dữ liệu đi kèm.

- **Điều kiện đầu:** cho vận tốc không phân kỳ $\mathbf{u}(\mathbf{x},0)=\mathbf{u}_0(\mathbf{x})$.
- **Thành không trượt:** vận tốc chất lưu bằng vận tốc thành. Với thành đứng yên, $\mathbf{u}=0$.
- **Biên vào:** cho profile vận tốc hoặc lưu lượng khối tương thích.
- **Biên ra:** thường cho áp suất và dùng điều kiện yếu cho vận tốc; nên đặt biên đủ xa vùng hồi lưu.
- **Biên tuần hoàn:** ghép các trường trên hai mặt tương ứng, hữu ích cho kênh lý tưởng và nhiễu loạn đồng nhất.
- **Biên trượt tự do hoặc đối xứng:** chặn vận tốc pháp tuyến và đặt ứng suất cắt tiếp tuyến bằng không.

Điều kiện biên của áp suất liên kết với điều kiện vận tốc; gán cả hai một cách tùy ý có thể làm bài toán bị thừa ràng buộc.

## Hai nghiệm chính xác giúp xây dựng trực giác

### Dòng Couette

Đặt chất lưu giữa hai tấm phẳng tại $y=0$ và $y=h$. Tấm dưới đứng yên, tấm trên chuyển động với vận tốc $U$. Với dòng ổn định, phát triển đầy đủ và không có gradient áp suất, $\mathbf{u}=(u(y),0,0)$ và

$$
\mu\frac{d^2u}{dy^2}=0.
$$

Áp dụng $u(0)=0$ và $u(h)=U$:

$$
\boxed{u(y)=U\frac{y}{h}.}
$$

Độ nhớt truyền động lượng từ tấm đang chuyển động vào chất lưu, tạo profile tuyến tính và ứng suất cắt không đổi $\tau_{xy}=\mu U/h$.

### Dòng Poiseuille phẳng

Giữ hai tấm tại $y=\pm h$ đứng yên và tạo dòng bằng gradient áp suất không đổi $dp/dx<0$. Phương trình rút gọn thành

$$
0=-\frac{dp}{dx}+\mu\frac{d^2u}{dy^2}.
$$

Với $u(-h)=u(h)=0$:

$$
\boxed{
u(y)=-\frac{1}{2\mu}\frac{dp}{dx}(h^2-y^2).
}
$$

Profile có dạng parabol. Vận tốc lớn nhất nằm tại đường tâm và vận tốc trung bình trên tiết diện là $\bar{u}=\tfrac{2}{3}u_{\max}$ đối với kênh phẳng này.

## Độ xoáy và động năng

Độ xoáy đo chuyển động quay cục bộ:

$$
\boldsymbol{\omega}=\nabla\times\mathbf{u}.
$$

Lấy toán tử xoáy của phương trình động lượng không nén được:

$$
\frac{\partial\boldsymbol{\omega}}{\partial t}
+(\mathbf{u}\cdot\nabla)\boldsymbol{\omega}
=(\boldsymbol{\omega}\cdot\nabla)\mathbf{u}
+\nu\nabla^2\boldsymbol{\omega}
+\nabla\times\mathbf{f}.
$$

Hạng tử $(\boldsymbol{\omega}\cdot\nabla)\mathbf{u}$ kéo giãn và nghiêng xoáy trong ba chiều. Nó triệt tiêu với dòng không nén được hai chiều thuần túy; đây là một lý do hành vi toán học của hệ 2D được kiểm soát tốt hơn.

Với biên tuần hoàn hoặc điều kiện suy giảm/không trượt thích hợp, cân bằng động năng là

$$
\frac{1}{2}\frac{d}{dt}\int_\Omega |\mathbf{u}|^2\,d\mathbf{x}
=-\nu\int_\Omega |\nabla\mathbf{u}|^2\,d\mathbf{x}
+\int_\Omega \mathbf{f}\cdot\mathbf{u}\,d\mathbf{x}.
$$

Đối lưu và áp suất phân phối lại năng lượng; độ nhớt tiêu tán nó. Đẳng thức này vừa đem lại trực giác vật lý, vừa là phép kiểm tra quan trọng cho bộ giải số.

## Giải hệ bằng phương pháp chiếu

Đa số hình học thực tế cần xấp xỉ số. Sai phân hữu hạn, thể tích hữu hạn, phần tử hữu hạn và phương pháp phổ rời rạc hóa không gian theo những cách khác nhau, nhưng mọi bộ giải không nén được đều phải ghép vận tốc với áp suất và duy trì phân kỳ gần bằng không.

Một phương pháp chiếu cơ bản tiến một bước thời gian qua ba giai đoạn. Trước hết, dự đoán vận tốc chưa dùng áp suất mới:

$$
\mathbf{u}^*=\mathbf{u}^n+\Delta t\left[
-(\mathbf{u}^n\cdot\nabla)\mathbf{u}^n
+\nu\nabla^2\mathbf{u}^n+\mathbf{f}^n
\right].
$$

Tiếp theo, giải phương trình Poisson cho áp suất:

$$
\nabla^2p^{n+1}=\frac{\rho}{\Delta t}\nabla\cdot\mathbf{u}^*.
$$

Cuối cùng, chiếu vận tốc lên không gian không phân kỳ:

$$
\mathbf{u}^{n+1}=\mathbf{u}^*-\frac{\Delta t}{\rho}\nabla p^{n+1}.
$$

Khung thuật toán khá ngắn:

~~~python
for step in range(num_steps):
    convection = advect(velocity)
    diffusion = viscosity * laplacian(velocity)
    predicted = velocity + dt * (-convection + diffusion + force)

    rhs = density / dt * divergence(predicted)
    pressure = solve_poisson(rhs, pressure_boundary_conditions)

    velocity = predicted - dt / density * gradient(pressure)
    velocity = apply_velocity_boundary_conditions(velocity)
~~~

Một bộ giải thực tế còn cần các toán tử gradient/phân kỳ rời rạc nhất quán, lược đồ đối lưu ổn định, bộ giải tuyến tính phù hợp, kiểm tra chất lượng lưới và điều kiện biên áp suất đúng. Với lược đồ tường minh, hai ước lượng hữu ích là

$$
\Delta t\lesssim C\frac{\Delta x}{U_{\max}},
\qquad
\Delta t\lesssim C_\nu\frac{\Delta x^2}{\nu},
$$

tương ứng với ổn định đối lưu và khuếch tán. Các hằng số phụ thuộc vào số chiều và cách rời rạc hóa.

## Bài toán tồn tại và tính trơn

Trong hai chiều, dữ liệu không nén được đủ chính quy dẫn tới nghiệm trơn toàn cục. Trong ba chiều, nghiệm yếu Leray toàn cục tồn tại, nhưng tính duy nhất và chính quy đầy đủ vẫn chưa được biết. Câu hỏi mở là: mọi điều kiện đầu trơn, năng lượng hữu hạn và không phân kỳ có luôn tạo ra nghiệm trơn với mọi thời gian, hay một kỳ dị có thể hình thành trong thời gian hữu hạn?

Đây là một trong các Bài toán Thiên niên kỷ của Clay Mathematics Institute. Một mô phỏng nhiễu loạn xuất hiện các thang rất nhỏ chưa phải bằng chứng cho kỳ dị toán học: cần phân biệt giới hạn độ phân giải, sai số rời rạc và sự khác nhau giữa nghiệm yếu với nghiệm cổ điển.

## Danh sách kiểm tra thực tiễn

Khi xây dựng hoặc đánh giá một mô hình dòng không nén được, hãy hỏi:

1. Giả định mật độ không đổi có hợp lý không, hay cần mô hình nén được/biến thiên mật độ?
2. Các giá trị đặc trưng $U$, $L$ và số Reynolds là bao nhiêu?
3. Điều kiện đầu và biên có tương thích và bảo toàn khối lượng không?
4. Lưới có phân giải được thành, lớp cắt và các thang nhiễu loạn cần thiết không?
5. Bước thời gian có thỏa các giới hạn ổn định đối lưu và khuếch tán không?
6. Vận tốc rời rạc có duy trì phân kỳ gần bằng không không?
7. Cân bằng khối lượng, động lượng và năng lượng có khép kín trong sai số cho phép không?
8. Kết quả đã được so sánh với nghiệm chính xác, benchmark hoặc nghiên cứu tinh chỉnh lưới chưa?

Phương trình rất gọn; một nghiệm đáng tin cậy thì không. Cơ học chất lưu tốt đòi hỏi kết hợp định luật bảo toàn với mô hình hóa, điều kiện biên, phương pháp số và kiểm chứng cẩn thận.

## Đọc thêm

- [Clay Mathematics Institute: Navier–Stokes Equation](https://www.claymath.org/millennium/navier-stokes-equation/) và [mô tả bài toán chính thức](https://www.claymath.org/wp-content/uploads/2022/06/navierstokes.pdf).
- [NASA Glenn: Navier–Stokes Equations](https://www.grc.nasa.gov/www/k-12/airplane/nseqs.html) cho dạng định luật bảo toàn dùng trong khí động học.
- A. J. Chorin, [*Numerical Solution of the Navier–Stokes Equations*](https://doi.org/10.1090/S0025-5718-1968-0242392-2), về ý tưởng phương pháp chiếu.

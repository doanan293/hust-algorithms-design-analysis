**ĐỀ BÀI SỐ 5 – ĐỀ BÀI TỐI ƯU PHỤC VỤ NGHIÊN CỨU  
CHO HỌC PHẦN DESIGN AND ANALYSIS OF ALGORITHMS**

> Trích đề số 5 từ danh sách 15 đề bài của học phần. Các mục 1, 3 và 4 là yêu cầu chung của học phần.

# 1. Tiêu chí thiết kế đề bài

Các đề bài dưới đây không chỉ yêu cầu cài đặt một thuật toán có sẵn. Mỗi đề cần được phát triển theo hướng có khoảng trống nghiên cứu, bộ benchmark đã được công bố, baseline mạnh, cơ chế thuật toán mới và khả năng mở rộng sang đa mục tiêu, động, online hoặc bất định.

- Câu hỏi nghiên cứu cụ thể.
- Mô hình hóa rõ input, output, mục tiêu và ràng buộc.
- Thuật toán baseline chuẩn và đề xuất ít nhất một cơ chế cải tiến có thể kiểm chứng.
- Phân tích độ phức tạp, tính khả thi, cận chất lượng hoặc expected-time khi phù hợp.
- Thực nghiệm: bộ dữ liệu benchmark đã công bố, seed ngẫu nhiên, lặp lại, kiểm định thống kê.
- Đánh giá objective, runtime, robustness, feasibility, scalability và sensitivity.

# 2. Đề bài được giao: đề số 5

| # | Đề bài nghiên cứu | Hướng đóng góp thuật toán | Dataset/benchmark công khai | Bài báo tham khảo gần và sát nhất |
|---|---|---|---|---|
| 5 | Joint UAV trajectory–relay–bandwidth allocation | Phân rã quỹ đạo, relay selection, bandwidth/power allocation để tối đa timely-alert probability và connectivity. | SNDlib; Topology Zoo + target từ RescueNet. | Huang et al., “Dynamic Dual-Antenna Time-Slot Allocation Protocol for UAV-Aided Relaying System Under Probabilistic LoS-Channel” (2025): time-slot, power, trajectory cho UAV relay. |

# 3. Lộ trình triển khai

| Thời gian | Yêu cầu tối thiểu | Bằng chứng nghiên cứu cần bổ sung |
|---|---|---|
| Tuần 1-2 | Mô hình + baseline + benchmark nhỏ | Phân tích complexity, objective và runtime |
| Tuần 3-6 | Thuật toán cải tiến + benchmark đầy đủ/ Trình bày ý tưởng | Ablation, sensitivity, 20–30 seed, kiểm định thống kê |
| Tuần 7-9 | Novel mechanism rõ ràng / Trình bày kết quả | So sánh SOTA, reproducibility package, case study |
| Tuần 10 | Nộp báo cáo | Abstract, Introduction, Related Work, Proposed Method, Experiment Results, Discussion, Conclusion |

# 4. Yêu cầu giao bài tập theo hai pha

Pha 1 – bắt buộc: cài đặt thuật toán baseline trên bộ dữ liệu benchmark đã công bố, có báo cáo kết quả và mã nguồn cài đặt.

Pha 2 – nghiên cứu: chọn ít nhất một hướng mở rộng:

- Đa mục tiêu.
- Tối ưu động/online.
- Bất định/stochastic/robust.
- Prediction-aware optimization.
- Learning-assisted heuristic.
- Decomposition/parallelization.
- Constraint repair.
- Approximation/guarantee.
- Optimization inspired by quantum/QUBO.
- Công bằng, rủi ro và độ tin cậy.

Báo cáo cần viết theo cấu trúc của bài nghiên cứu khoa học như một bài báo: Abstract, Introduction, Related Work, Proposed Method, Experiment Results, Discussion, Conclusion

# Đề bài và định hướng: Joint UAV trajectory–relay–bandwidth allocation

> Cập nhật 10/09/2026 sau khi đối chiếu đề số 5 của học phần Design and Analysis of Algorithms. Người thực hiện đã duyệt khung một UAV di động kết hợp các relay mặt đất cố định. Các chi tiết mô hình dưới đây là phương án triển khai đề xuất để kiểm chứng; chưa phải kết quả nghiên cứu hoặc bằng chứng về tính mới. Bản rà soát này bổ sung: phiên bản v0 cho Pha 1 (Mục 3.4), baseline kế thừa trực tiếp cấu trúc BCD của bài Huang (Mục 9.1), cận trên LP và gap (Mục 9.4), connectivity làm mục tiêu phụ theo thứ tự từ điển (Mục 8.7), quy ước tỷ lệ không gian cố định (Mục 7).

## 1. Căn cứ và yêu cầu môn học

Tài liệu quy định đề bài: [Đề số 5 – hướng nghiên cứu bài tập môn học](references/huong-nghien-cuu-de-so-5.md) (trích từ danh sách 15 đề của học phần):

> Phân rã quỹ đạo, relay selection, bandwidth/power allocation để tối đa timely-alert probability và connectivity.

Dataset/benchmark được nêu: **SNDlib; Topology Zoo + target từ RescueNet**.

Bài báo tham chiếu chính: **Huang, P.; Lin, J.; Liu, T.; Ning, J.; Luo, J.; Duo, B. (2025). Dynamic Dual-Antenna Time-Slot Allocation Protocol for UAV-Aided Relaying System Under Probabilistic LoS-Channel. Sensors, 25, 7443.** [Bản Markdown toàn văn](references/sensors-25-07443-v2.md) (chuyển từ PDF gốc, giấy phép CC BY 4.0; hình ở `references/sensors-25-07443-v2-figures/`), [DOI](https://doi.org/10.3390/s25247443).

Yêu cầu môn học chi phối lựa chọn bài toán; bài Huang cung cấp cơ sở chuyên môn và thuật toán. Không đồng nhất mục tiêu max–min rate của bài Huang với mục tiêu cảnh báo đúng hạn của đề số 5.

### 1.1. Sản phẩm bắt buộc

- Pha 1: cài đặt baseline trên benchmark đã công bố, có mã nguồn và báo cáo kết quả.
- Pha 2: ít nhất một hướng mở rộng, có cơ chế cải tiến kiểm chứng được.
- Nêu rõ input, output, objective, constraints và câu hỏi nghiên cứu.
- Phân tích độ phức tạp, tính khả thi và cận chất lượng khi có căn cứ.
- Thực nghiệm lặp lại với **20–30 seed**, ablation, sensitivity và kiểm định thống kê.
- Đánh giá objective, runtime, robustness, feasibility, scalability và sensitivity.
- Báo cáo theo cấu trúc bài nghiên cứu; có reproducibility package và case study.
- Lộ trình mẫu của môn học kéo dài 10 tuần. Đây là khung lập kế hoạch, không phải thời gian còn lại đã được xác nhận.

### 1.2. Điều kiện thực hiện

Nghiên cứu hoàn toàn bằng mô phỏng trên máy tính. Không yêu cầu UAV thật, thiết bị vô tuyến hoặc bay thử. Dùng dữ liệu công khai kết hợp tham số vật lý và giả định kịch bản có nguồn gốc rõ ràng. Không cần huấn luyện mô hình thị giác để dùng mask RescueNet.

## 2. Tên đề tài và phát biểu bài toán

**Tên tiếng Việt:** Đồng tối ưu quỹ đạo UAV, lựa chọn relay và phân bổ băng thông nhằm nâng cao xác suất truyền cảnh báo đúng hạn trong mạng cứu hộ.

**Tên tiếng Anh dự kiến:** Joint UAV Trajectory, Relay Selection, and Bandwidth Allocation for Deadline-Aware Alert Delivery in Disaster Networks.

Sau thiên tai, một số liên kết mặt đất bị mất hoặc suy giảm. Các điểm phục vụ phát sinh cảnh báo có thời điểm xuất hiện, kích thước và hạn giao. Một UAV làm relay di động để hỗ trợ mạng relay cố định đưa dữ liệu đến trung tâm cứu hộ.

**Bài toán cần giải:** đồng quyết định đường bay của UAV, relay/đường chuyển tiếp cho từng cảnh báo và băng thông trên các liên kết để tăng tỷ lệ cảnh báo được giao đủ trước hạn dưới kênh bất định, trong giới hạn tài nguyên và di chuyển. Connectivity theo thời gian đến trung tâm là mục tiêu phụ, xét theo thứ tự từ điển sau tỷ lệ đúng hạn.

Một relay gần nguồn chưa chắc tốt: nó có thể thiếu backhaul, bị quá tải hoặc nằm trên đường chuyển tiếp nhiều chặng khiến cảnh báo trễ hạn. Tương tự, UAV bay đến điểm có kênh mạnh nhất chưa chắc phục vụ được nhiều cảnh báo đúng hạn nhất. Đó là sự phụ thuộc giữa các khối quyết định cần nghiên cứu.

## 3. Phạm vi đã chọn và phạm vi mở rộng

### 3.1. Mô hình chính

- Một UAV, nhiều relay mặt đất cố định và một trung tâm cứu hộ.
- Cảnh báo truyền có hướng từ điểm phục vụ về trung tâm; không mặc định trao đổi hai chiều như bài Huang.
- UAV thực hiện decode-and-forward với buffer.
- UAV có độ cao cố định, điểm đầu/cuối, giới hạn tốc độ và vùng cấm bay nếu scenario có.
- Thiết kế offline theo workload và thống kê kênh biết trước. Pha 1 thiết kế trên tốc độ kỳ vọng (surrogate xác định, $M=1$); đánh giá cuối luôn trên realization kênh độc lập.
- Lựa chọn relay là quyết định thực sự giữa nhiều relay mặt đất hoặc UAV.
- Băng thông vô tuyến là biến tối ưu; công suất hoạt động cố định theo liên kết trong phiên bản đầu.
- Liên kết mặt đất/backhaul có dung lượng hữu hạn và chia sẻ giữa các luồng.
- Bất định kênh UAV được xét ở bước đánh giá trong Pha 1 và đưa vào bước thiết kế ở Pha 2 (Mục 5.2); trạng thái hỏng mạng được sinh trước mỗi scenario.
- Mục tiêu chính là tỷ lệ cảnh báo đúng hạn; connectivity theo thời gian là mục tiêu phụ theo thứ tự từ điển (Mục 8.7).

### 3.2. Quy ước liên kết để giữ mô hình vừa sức

Điểm nguồn có thể gửi đến relay mặt đất đủ điều kiện hoặc UAV. UAV chỉ nhận từ nguồn và chuyển xuống relay mặt đất/trung tâm; không cho phép lặp nguồn–UAV–mặt đất–UAV tùy ý. Sau khi vào mạng relay mặt đất, dữ liệu đi đến trung tâm qua đường backhaul được chọn.

Phiên bản đầu cho phép **store-carry-forward**: UAV có thể nhận ở vị trí này và chuyển tiếp ở slot sau. Vì vậy, khả năng giao dữ liệu phải được xét trên mạng có thời gian, không bắt buộc tồn tại đường kết nối đồng thời ở mọi slot.

### 3.3. Những phần chưa đưa vào lõi

Nhiều UAV, tránh va chạm, học tăng cường, huấn luyện segmentation, tối ưu công suất liên tục, hỏng mạng online và bảo đảm rủi ro phân phối chưa biết là mở rộng sau. Không thêm đồng thời chỉ để tăng số biến. Công suất được khảo sát như tham số trong phiên bản đầu.

### 3.4. Phiên bản v0 cho Pha 1

v0 là bài toán nhỏ nhất vẫn đúng đề số 5 và chạy được trong tuần 1–2. Mọi tính năng ngoài danh sách dưới đây chỉ được thêm sau khi v0 có số liệu.

| Thành phần | v0 | Thêm sau v0 |
|---|---|---|
| Mạng | Một topology Topology Zoo cỡ nhỏ (15–40 nút, có tọa độ, liên thông), chọn theo quy tắc công bố; relay là tập con nút; một trung tâm | Nhiều topology; SNDlib demand đầy đủ |
| Alert | 20–60 alert; cường độ từ demand SNDlib hoặc quy tắc công bố; release/deadline sinh có seed | Priority theo thiệt hại |
| Kênh | PrLoS Huang, thiết kế trên tốc độ kỳ vọng ($M=1$); đánh giá trên 20–30 realization | SAA trong thiết kế (Pha 2) |
| UAV | Một UAV, độ cao cố định, không vùng cấm bay | Vùng cấm bay |
| Giao thức | Hai pha cố định, phổ trực giao, công suất cố định, half-duplex | Tối ưu $\tau$, $p_e$ |
| Path | Candidate paths $K\le 3$, một path mỗi alert, store-carry-forward | $K$ lớn hơn |
| Phương pháp | Evaluator + test, B0, B1, cận trên LP trên quỹ đạo B1 (Mục 9.4) | B2, B3, P |
| Dữ liệu ảnh | Không dùng RescueNet | Case study RescueNet ở tuần 7–9 |

Điều kiện hoàn thành v0: evaluator qua bộ test ở Mục 11; bảng objective, gap so với cận LP và runtime trên ít nhất 10 scenario với 20–30 seed đánh giá.

## 4. Kế thừa bài Huang và phần thay đổi

| Thành phần | Bài Huang | Đề tài này |
|---|---|---|
| Mạng | Một UAV phục vụ nhiều cặp người dùng | Một UAV hỗ trợ mạng relay cố định và trung tâm |
| Lưu lượng | Trao đổi hai chiều, tốc độ trung bình | Cảnh báo có release time, kích thước và deadline |
| Mục tiêu | Max–min average rate | Kỳ vọng tỷ lệ cảnh báo giao đúng hạn |
| Relay | UAV đã được xác định | Chọn relay/đường chuyển tiếp |
| Tài nguyên | RSF, công suất, quỹ đạo | Băng thông, relay/path, quỹ đạo; công suất cố định ban đầu |
| Kênh UAV | PrLoS | Kế thừa làm mô hình thống kê |
| Buffer | Nhân quả thông tin | Kế thừa, thêm deadline và đường nhiều chặng |
| Giao thức | Dual-antenna full-duplex DDATSAP | Half-duplex, phổ trực giao theo mô hình Mục 8 |
| Phương pháp | BCD–SCA | B2 là BCD cùng cấu trúc ba khối của bài Huang áp lên bài toán mới; P thêm sửa lịch theo slack; mọi kết quả kèm gap so với cận trên LP |

**Không tuyên bố tái lập nguyên trạng DDATSAP.** Bản đơn giản hóa dùng để nghiên cứu joint optimization theo yêu cầu môn học. Nếu tái lập bài Huang, thực hiện thành thí nghiệm riêng với đúng giao thức, công suất và đơn vị; không so trực tiếp giá trị objective khác loại.

Các mục cần đọc kỹ trong [bản Markdown của bài báo](references/sensors-25-07443-v2.md): [Mục 3](references/sensors-25-07443-v2.md#3-system-model) về kênh/giao thức/nhân quả; [Mục 4](references/sensors-25-07443-v2.md#4-problem-formulation) về bài toán; [Mục 5](references/sensors-25-07443-v2.md#5-proposed-algorithm) về phân rã/xấp xỉ; [Mục 6](references/sensors-25-07443-v2.md#6-numerical-results) và Table 4 (tham số mô phỏng) về đánh giá. Bảng ký hiệu nằm ở [Mục 2](references/sensors-25-07443-v2.md#2-list-of-mathematical-symbols-and-variables) (Table 2). Rà soát ký hiệu slot, noise PSD so với noise power, băng thông, ngân sách công suất tổng và ánh xạ uplink/downlink trước khi kế thừa công thức.

## 5. Câu hỏi nghiên cứu và cơ chế cải tiến

**RQ1:** Tối ưu đồng thời quỹ đạo, relay và băng thông có tăng tỷ lệ cảnh báo đúng hạn so với tối ưu từng phần không?

**RQ2:** Cơ chế nhận diện nguy cơ trễ hạn và sửa lựa chọn relay/tài nguyên mang lại lợi ích nào so với phân rã thông thường?

**RQ3:** UAV giúp khôi phục khả năng chuyển dữ liệu đến trung tâm trong những dạng hỏng mạng và phân bố target nào?

**RQ4:** Thuật toán suy giảm ra sao khi tăng quy mô, siết deadline, giảm phổ hoặc sai lệch mô hình kênh?

**RQ5:** Kết luận có bền vững trên nhiều seed, nhiều topology và nhiều quy tắc tạo target không?

**Thứ tự ưu tiên:** RQ1 và RQ2 là câu hỏi lõi, quyết định kết luận chính và cần thí nghiệm riêng (so sánh B0/B1/B2/B3/P và ablation). RQ3–RQ5 được trả lời bằng các phân tích sensitivity, scalability và robustness ở Mục 10 trên cùng bộ kết quả; không tạo thí nghiệm mới nếu thời gian hạn chế.

### 5.1. Cơ chế đề xuất: sửa lịch theo nguy cơ trễ hạn

Tên mô tả tạm thời: **deadline-aware relay and bandwidth repair**.

Cơ chế gồm đúng **hai thao tác**, để ablation quy được lợi ích về từng thao tác. Sau mỗi vòng phân rã:

1. Ước lượng slack của từng alert: thời gian giao dự kiến theo path hiện tại, backlog, dung lượng dự kiến, so với deadline.
2. Xác định alert nguy cơ: slack âm hoặc dưới ngưỡng, hoặc bị nghẽn ở access/backhaul.
3. **Thao tác 1 — chuyển băng thông:** dời băng thông từ luồng còn dư slack sang alert nguy cơ trong cùng pha và cùng slot.
4. **Thao tác 2 — đổi relay/path:** thử path khác trong candidate set cho alert nguy cơ, ưu tiên path giảm số chặng nghẽn.
5. Giải lại bài toán băng thông (Mục 9.2) để kiểm tra cạnh tranh tài nguyên và nhân quả trên toàn hệ.
6. Chỉ nhận ứng viên theo objective từ điển cố định (Mục 8.7); không ưu tiên một alert bất khả thi làm mất nhiều alert khả thi.

Quỹ đạo không thuộc cơ chế repair; khối local search quỹ đạo trong B2 xử lý. Ghép đổi relay với sửa quỹ đạo là biến thể mở rộng, chỉ xét khi ablation cho thấy hai thao tác trên bị kẹt. Các bước này là **giả thuyết thuật toán**, phải được đo bằng ablation và rà soát literature; chưa được gọi là mới chỉ vì đặt tên.

### 5.2. Hướng mở rộng môn học đã chọn

Hướng lõi của Pha 2 là **decomposition và constraint repair** (Mục 5.1). Hướng thứ hai, chỉ làm ở tuần 7–9 nếu còn thời gian, là **bất định kênh trong thiết kế**: thay surrogate kỳ vọng bằng SAA trên $M$ realization thiết kế (Mục 8.7); evaluator đã có sẵn nên chi phí cài đặt thêm thấp, và so sánh "thiết kế kỳ vọng" với "thiết kế SAA" trên cùng tập đánh giá là một kết quả rõ ràng. Không bắt buộc dùng tất cả hướng mở rộng trong tài liệu môn học.

## 6. Dataset và cách tải

### 6.1. SNDlib

Nguồn chính: [SNDlib tại ZIB](https://sndlib.zib.de/home.action).

Dùng topology và demand khi instance cung cấp, kiểm tra đơn vị và ý nghĩa từng trường. Giá trị thiết kế/mô-đun dung lượng không tự động là dung lượng vận hành hiện hữu. Nếu cần lựa chọn mức dung lượng, ghi quy tắc và đánh dấu phần cấu hình bổ sung.

Tại lần khảo sát trước, công cụ chưa truy cập được trang gốc; chưa xác minh được URL tải trực tiếp. Đầu việc thu thập phải thử lại nguồn chính, ghi nhận kết quả, rồi mới xem xét mirror có nguồn gốc đối chiếu được. Không ghi dataset đã tải khi mới có liên kết.

Demand SNDlib dùng làm nguồn workload về cường độ/kích thước tương đối. Nếu chuyển demand nhiều đích thành cảnh báo về một trung tâm, phải công bố phép biến đổi; dữ liệu sau chuyển đổi không còn là demand nguyên bản.

### 6.2. Topology Zoo

Nguồn: [website dự án](https://topology-zoo.org/) và [repository](https://github.com/sk2/topologyzoo).

Có thể tải Code → Download ZIP hoặc:

~~~bash
git clone --depth 1 https://github.com/sk2/topologyzoo.git data/raw/topology-zoo
~~~

Lưu commit sau khi tải. Kiểm tra tọa độ, graph đa cạnh, kết nối và trường capacity từng instance. Không mặc định có demand/capacity đầy đủ. Các trường thiếu được cấu hình công khai.

Cạnh backbone đại diện liên kết mặt đất/backhaul trong mô hình; không được coi là cạnh vô tuyến UAV. Mô hình UAV dùng tọa độ và kênh riêng.

### 6.3. RescueNet

- [Repository tác giả](https://github.com/BinaLab/RescueNet-A-High-Resolution-Post-Disaster-UAV-Dataset-for-Semantic-Segmentation).
- [Collection Figshare](https://springernature.figshare.com/collections/RescueNet_A_High_Resolution_UAV_Semantic_Segmentation_Benchmark_Dataset_for_Natural_Disaster_Damage_Assessment/6647354/1).

Tải ảnh và mask tương ứng từ từng mục dữ liệu; repository tác giả còn liên kết Dropbox. Clone mã nguồn không tải ảnh dataset. Kiểm tra dung lượng trước khi tải toàn bộ; pilot có thể dùng một tập con với manifest.

Dùng mask công trình hư hại để tạo điểm phục vụ giả định. Mask không cung cấp vị trí nạn nhân, deadline, traffic, chiều cao vật cản hoặc channel measurement.

Kiểm tra license bản dữ liệu thực tải; README tác giả ghi CC BY-NC-ND cho dataset, không đồng nhất với license mã nguồn. Ưu tiên công bố manifest và mã chuyển đổi; kiểm tra điều kiện trước khi phân phối ảnh/mask đã biến đổi.

RescueNet chỉ dùng cho case study ở tuần 7–9 với một tập con ảnh nhỏ; không nằm trong v0 và không chặn tiến độ Pha 1.

### 6.4. Vai trò của ba nguồn

| Nguồn | Vai trò | Giả định cần bổ sung |
|---|---|---|
| SNDlib | Cấu trúc mạng và workload có thông tin demand | Chọn trung tâm, chuyển demand thành alert, deadline, failure |
| Topology Zoo | Đa dạng cấu trúc relay/backhaul | Capacity/demand thiếu, chuẩn hóa tọa độ, failure |
| RescueNet | Hình học target từ vùng hư hại | Tỷ lệ mét, thiết bị phát, kích thước alert, deadline |

Hai nhóm đánh giá: mạng từ SNDlib/Topology Zoo; case study target RescueNet kết hợp một topology đã chuẩn hóa. Các nguồn không cùng địa lý. Mọi kết hợp là **kịch bản tổng hợp từ dữ liệu công khai**, không phải mạng cứu hộ được đo thực địa.

## 7. Quy trình xây scenario

1. Lưu URL, ngày tải, phiên bản, license, dung lượng và SHA-256; giữ raw bất biến.
2. Chọn graph/subgraph, relay và trung tâm theo quy tắc cố định.
3. Chiếu tọa độ địa lý sang hệ mét rồi chuẩn hóa bounding box về vùng $L_{\mathrm{box}}\times L_{\mathrm{box}}$, mặc định $L_{\mathrm{box}}=2$ km, cùng tỷ lệ trên hai trục, ghi hệ số. Chọn $T$ và $V_{\max}$ sao cho $V_{\max}T\ge 2\sqrt{2}\,L_{\mathrm{box}}$ (UAV đi được ít nhất hai đường chéo vùng). Khảo sát sensitivity với $L_{\mathrm{box}}\in\{1,2,4\}$ km. Mạng backbone gốc có quy mô hàng trăm km nên sau thu nhỏ chỉ còn mượn cấu trúc; phải ghi rõ đây là scenario tổng hợp. Không dùng layout đồ thị tùy ý như tọa độ địa lý thật.
4. Với RescueNet, xác minh class ID, tìm vùng liên thông, lọc nhiễu, chọn điểm đại diện trong vùng và gộp điểm gần nhau. Lưu ánh xạ về ảnh/mask/vùng.
5. Kiểm tra georeferencing/GSD. Khi thiếu, dùng hệ số pixel–mét giả định và khảo sát sensitivity. Không ghép ảnh khác nhau thành bản đồ liên tục khi chưa có căn cứ.
6. Tạo alert: nguồn, release time, số bit, deadline; mặc định trọng số bằng nhau. Priority theo thiệt hại chỉ là biến thể có công bố quy tắc.
7. Tạo tình trạng hỏng/giảm dung lượng trước nhiệm vụ với seed; giữ cả scenario còn kết nối và mất kết nối. Road-blocked trong mask không mặc nhiên là no-fly zone.
8. Cấu hình UAV, kênh, phổ, thời gian, candidate paths và seed.
9. Tách tập phát triển, kịch bản thiết kế kênh và tập đánh giá độc lập.

Mỗi scenario chứa đủ thông tin để tái tạo topology, tọa độ, alerts, failure, channel, UAV và các phép chuẩn hóa. Lưu cả lý do loại bỏ dữ liệu. Với ảnh gần trùng/vùng gần nhau, không coi nhiều ảnh là các mẫu độc lập nếu metadata không hỗ trợ.

## 8. Mô hình toán đề xuất

### 8.0. Tóm tắt input, output, objective, constraints

| Thành phần | Nội dung |
|---|---|
| Input | Graph $G=(V,E)$ với capacity backhaul $C_e[n]$ và trạng thái hỏng; tọa độ nút đã chuẩn hóa; trung tâm $c$; tập alert $A$ với $(s_a, r_a, d_a, L_a)$; tham số UAV $(H, V_{\max}, q_s, q_t)$; tham số kênh PrLoS $(a, b, \mu, \beta_0, \alpha_L, \alpha_N, N_0)$; $B_{\mathrm{tot}}$, $p_e$, $\tau$, $T$, $\Delta$; số candidate paths $K$ |
| Output | Quỹ đạo $q[0],\dots,q[N]$; lựa chọn path/relay $x_{a,p}$; băng thông $b_e[n]$; lịch flow $f$ |
| Objective | Thứ tự từ điển: (1) tối đa tỷ lệ alert giao đúng hạn; (2) tối đa connectivity theo thời gian; (3) tối thiểu tổng lateness (Mục 8.7) |
| Constraints | Di chuyển (8.3); phổ hai pha (8.4); capacity vô tuyến và backhaul (8.5–8.6); bảo toàn flow, buffer không âm, nhân quả slot, release/deadline (8.6); một path mỗi alert (8.2) |
| Bất định | Trạng thái LoS/NLoS theo slot và theo vị trí UAV; Pha 1 thiết kế trên kỳ vọng, đánh giá trên realization (8.8) |

### 8.1. Tập và biến

- $G=(V,E)$: mạng relay mặt đất; $c\in V$ là trung tâm.
- $S$: các điểm nguồn cảnh báo; $U$ là UAV.
- $A$: tập cảnh báo; mỗi $a$ có nguồn $s_a$, release $r_a$, deadline $d_a$, kích thước $L_a$ bit.
- $T=N\Delta$: thời gian nhiệm vụ; $q[0],\dots,q[N]$ là waypoint ngang của UAV.
- $x_{a,p}\in\{0,1\}$: chọn candidate path $p$ cho alert $a$; cho phép không chọn khi không thể phục vụ.
- $b_e[n]\ge 0$: băng thông hoạt động trên liên kết vô tuyến $e$ trong slot $n$.
- $f_{a,e,n,\omega}\ge 0$: số bit alert $a$ truyền trên $e$ trong realization $\omega$.
- $z_{a,\omega}\in\{0,1\}$: alert $a$ được giao đủ trước hạn trong realization $\omega$.
- Buffer theo alert, nút và slot; các đại lượng này phải được kiểm tra độc lập sau giải.

Không được cộng cùng một bit nhiều lần do đường vòng hoặc duplicate delivery. Phiên bản đầu mỗi alert dùng một candidate path; không chia alert sang nhiều relay/path để đơn giản hóa việc chọn relay.

### 8.2. Candidate paths và relay selection

Tạo tối đa $K$ đường đơn cho mỗi alert:

- Nguồn → relay mặt đất → đường backhaul → trung tâm.
- Nguồn → UAV → relay mặt đất hoặc trung tâm → backhaul nếu có.

Candidate path được sinh từ graph tĩnh và vị trí, nhưng khả năng truyền các cạnh UAV phụ thuộc slot. Cho phép chờ ở buffer. Chọn path gồm cả chọn relay đầu vào/relay xuống của UAV; giữ $K$ hữu hạn để kiểm soát chi phí.

Cạnh được chọn chưa bảo đảm có thể giao trước hạn. Việc xác nhận phải dựa trên flow theo thời gian và capacity. Tập candidate paths hạn chế miền tìm kiếm; mọi cận/nghiệm tham chiếu phải nêu hạn chế này.

### 8.3. Di chuyển

$$
q[0]=q_s,\quad q[N]=q_t,\quad
\|q[n]-q[n-1]\|_2\le V_{\max}\Delta.
$$

Độ cao $H$ cố định. Dùng vị trí trung điểm mỗi đoạn để xấp xỉ kênh trong slot, rồi kiểm tra độ nhạy theo $\Delta$. Vùng cấm bay phải được kiểm tra trên cả đoạn, không chỉ waypoint; dùng hành lang an toàn hoặc kiểm tra đoạn–vùng trong bước sửa quỹ đạo.

### 8.4. Giao thức và công suất

Mô hình đầu dùng hai pha cố định trong mỗi slot:

- Pha access, tỷ lệ $\tau=0.5$: nguồn gửi đến relay mặt đất hoặc UAV.
- Pha UAV downlink, tỷ lệ $1-\tau$: UAV chuyển đến relay mặt đất/trung tâm.
- Mạng backhaul dùng tài nguyên riêng với capacity hữu hạn; không trừ capacity này lần nữa vào phổ UAV/access.
- Các liên kết vô tuyến trong cùng pha dùng phổ trực giao, không gian tái sử dụng phổ chưa được xét.

$$
\sum_{e\in E_{\mathrm{access}}}b_e[n]\le B_{\mathrm{tot}},
\quad
\sum_{e\in E_{\mathrm{UAV-down}}}b_e[n]\le B_{\mathrm{tot}}.
$$

Hai pha tái sử dụng cùng phổ ở thời gian khác nhau. UAV half-duplex; không đưa tự nhiễu full-duplex vào công thức này.

$p_e$ là công suất hoạt động cố định, chọn sao cho tổng công suất các liên kết có thể cùng hoạt động không vượt ngân sách từng thiết bị. Giữ ngân sách tổng khi tăng số nguồn/relay; ghi rõ nếu cách đặt $p_e$ bảo thủ. Tối ưu $p_e$ hoặc $\tau$ là mở rộng, không phải phần lõi ban đầu.

### 8.5. Kênh và capacity

Kế thừa PrLoS của bài Huang (công thức (5), (14)–(17) trong [Mục 3](references/sensors-25-07443-v2.md#3-system-model)) cho các liên kết UAV:

$$
d_k[n]=\sqrt{\|\bar q[n]-w_k\|^2+H^2},\quad
\theta_k[n]=\frac{180}{\pi}\operatorname{atan2}(H,\|\bar q[n]-w_k\|),
$$

$$
P_k^L[n]=\frac{1}{1+a\exp[-b(\theta_k[n]-a)]},
\quad g_k^L[n]=\beta_0d_k[n]^{-\alpha_L},
\quad g_k^N[n]=\mu\beta_0d_k[n]^{-\alpha_N}.
$$

$N_0$ dùng W/Hz; gain và công suất dùng thang tuyến tính. Capacity bit/s trong pha:

$$
R_{e,n,\omega}=b_e[n]\log_2
\left(1+\frac{p_eg_{e,n,\omega}(q)}{N_0b_e[n]}\right).
$$

Số bit slot tối đa là $\Delta t_e R$, với $t_e=\tau$ hoặc $1-\tau$. Giá trị tại $b=0$ lấy bằng 0 theo giới hạn. Với cạnh backhaul, số bit tối đa là $\Delta C_e[n]$. Mọi luồng dùng cạnh đó chia sẻ cùng ngân sách.

Kênh access mặt đất cần mô hình path loss riêng hoặc capacity đã cấu hình, không tự dùng PrLoS UAV cho mặt đất. Ngưỡng range/SNR để tạo cạnh nếu sử dụng phải ghi rõ và có sensitivity.

Tốc độ kỳ vọng là tổng theo trạng thái LoS/NLoS của tốc độ, không phải log của gain trung bình. Trong tối ưu quỹ đạo có thể dùng tốc độ kỳ vọng làm surrogate; đánh giá xác suất cuối dùng realization thực.

### 8.6. Flow, buffer và deadline

$$
\sum_a f_{a,e,n,\omega}\le \Delta t_eR_{e,n,\omega}
$$

cho cạnh vô tuyến, tương tự $\Delta C$ cho backhaul.

Dữ liệu xuất khỏi một relay trong slot $n$ không vượt buffer của relay ở đầu slot. Dữ liệu nhận ở $n$ chỉ được chuyển tiếp từ $n+1$. Đây là quy ước độ trễ slot bảo thủ và thống nhất trên mọi phương pháp.

Nguồn chỉ có $L_a$ bit từ $r_a$; không phát trước release. Flow chỉ đi trên path đã chọn. Buffer cập nhật bằng nhận trừ phát và luôn không âm; không được tự sinh dữ liệu.

$$
\mathrm{Delivered}_{a,\omega}(d_a)\ge L_a z_{a,\omega}.
$$

Chỉ bit tới trung tâm trước deadline được tính vào điều kiện thành công. Dữ liệu đến trễ hoặc mới tới relay không được tính là cảnh báo đúng hạn.

### 8.7. Objective và connectivity

**Objective thiết kế Pha 1** dùng tốc độ kỳ vọng (Mục 8.5), nên $z_a$ xác định:

$$
\max\ \hat P_{\mathrm{timely}}=\frac{1}{|A|}\sum_{a\in A}z_a.
$$

**Objective đánh giá** (và objective thiết kế SAA ở Pha 2) trên $M$ realization:

$$
\hat P_{\mathrm{timely}}
=\frac{1}{M|A|}\sum_{\omega=1}^{M}\sum_{a\in A}z_{a,\omega}.
$$

Đây là tỷ lệ thành công trung bình trên alerts và realizations, không phải xác suất tất cả cảnh báo cùng thành công. Dùng bộ seed đánh giá mới để ước lượng kết quả cuối.

**Connectivity** được định nghĩa trên time-expanded graph, kể cả chờ ở buffer và store-carry-forward:

$$
\mathrm{Conn}=\frac{1}{|S|}\sum_{s\in S}
\mathbf{1}\big[\, s \leadsto c \ \text{in}\ [0,T] \,\big],
$$

trong đó $s \leadsto c$ nghĩa là tồn tại đường theo thời gian từ $s$ đến trung tâm $c$ trong khoảng $[0,T]$.

Connectivity được đo theo hai mức:

1. Có đường khả dụng theo thời gian đến trung tâm ($\mathrm{Conn}$ ở trên).
2. Có đủ dịch vụ để giao $L_a$ trước $d_a$, có xét cạnh tranh tài nguyên ($\hat P_{\mathrm{timely}}$).

Mức 1 không thay thế mức 2. Không yêu cầu toàn graph kết nối mọi thời điểm. Báo thêm thời gian mất kết nối và cải thiện so với không có UAV.

**Objective tổng hợp** theo thứ tự từ điển, dùng thống nhất cho mọi phương pháp khi so sánh và chấp nhận nghiệm:

$$
\big(\hat P_{\mathrm{timely}},\ \mathrm{Conn},\ -\textstyle\sum_a \mathrm{lateness}_a\big).
$$

Cách này đưa "connectivity" của đề số 5 vào objective mà không cần trọng số tùy ý. Nếu cần khảo sát trade-off, dùng ε-constraint trên $\mathrm{Conn}$ như một mở rộng đa mục tiêu, không đổi objective giữa các phương pháp để tạo lợi thế.

### 8.8. Offline, bất định và thông tin được phép dùng

$q$, $x$ và kế hoạch bandwidth được thiết kế từ workload cùng thống kê kênh (Pha 1: tốc độ kỳ vọng) hoặc kịch bản kênh thiết kế (Pha 2: SAA); không dùng realization đánh giá tương lai. Trong mô phỏng, dịch vụ thực tế bị giới hạn bởi capacity và buffer hiện tại.

Dispatcher chung cho các phương pháp có thể dùng EDF trong từng queue và bỏ alert hết hạn. Khi cùng release/deadline, dùng ID cố định để phá hòa. Nếu dùng flow recourse biết trước toàn bộ realization trong một bài toán SAA, phải ghi đó là đánh giá lạc quan/upper-bound có thông tin tương lai, không coi là policy triển khai.

Xác suất LoS phụ thuộc quỹ đạo: không lấy mẫu trạng thái tại một quỹ đạo rồi giữ nguyên phân phối cho mọi quỹ đạo khác. Có thể dùng cùng số ngẫu nhiên Uniform và ngưỡng $\mathrm{PLoS}(q)$ khi so sánh ứng viên. Cách này có thể làm objective mẫu gián đoạn; không mặc định SCA xử lý trực tiếp được.

## 9. Phương pháp triển khai

### 9.1. Baseline bắt buộc

- **B0 — không UAV:** cùng mạng hỏng, relay/path đơn giản, EDF và chia đều phổ giữa liên kết có backlog.
- **B1 — UAV quỹ đạo cố định:** đường khả thi qua vùng phục vụ, chọn path theo delay ước tính, EDF và chia đều phổ.
- **B2 — BCD kiểu Huang:** kế thừa đúng cấu trúc phân rã ba khối của bài Huang ([Mục 5 của bài báo](references/sensors-25-07443-v2.md#5-proposed-algorithm)) áp lên bài toán này: luân phiên khối relay/path, khối băng thông và khối quỹ đạo, mỗi khối tối ưu với hai khối kia cố định, dừng theo cùng tiêu chí; không có repair. Khối quỹ đạo dùng local search ở phiên bản đầu, SCA là biến thể nâng cao (Mục 9.2).
- **B3 — objective của Huang:** cùng cấu trúc B2 nhưng tối đa max–min tốc độ giao trung bình theo alert như bài Huang, rồi đánh giá bằng tỷ lệ đúng hạn. B3 đo trực tiếp chi phí của việc dùng sai objective, và là cầu nối công khai giữa bài tham chiếu và đề tài.
- **P — phương pháp đề xuất:** B2 cộng cơ chế sửa lịch hai thao tác ở Mục 5.1.
- **UB — cận trên LP:** không phải phương pháp, nhưng được báo trong mọi bảng để đo gap (Mục 9.4).
- Một baseline mạnh từ literature phù hợp objective và thông tin đầu vào phải được lựa chọn sau tổng quan; B0/B1 không đủ cho kết luận hơn SOTA.

B0 đo giá trị bổ sung UAV; B1/B2/B3/P phải giữ cùng ngân sách phần cứng, cùng candidate set và cùng dispatcher để đo lợi ích thuật toán.

### 9.2. Các khối

**Relay/path:** thử hoán đổi có giới hạn trong candidate set, dùng chi phí có backlog/deadline thay vì khoảng cách đơn thuần. Bài nhỏ có thể giải biến nhị phân bằng solver làm tham chiếu.

**Bandwidth:** với quỹ đạo và công suất cố định, capacity $b\log(1+c/b)$ lõm. Nhưng tối đa số alert hoàn thành có biến nhị phân vẫn không phải bài toán lồi. Có thể giải relaxation $z\in[0,1]$ cho đề xuất tài nguyên rồi làm tròn/sửa và đánh giá lại; hoặc giải bài lồi cho một tập alert đã chọn. Không gọi toàn khối là convex nếu còn lựa chọn hoàn thành nhị phân.

**Quỹ đạo:** phiên bản đầu dùng local search các waypoint/đoạn với kiểm tra tốc độ và vùng cấm. Nếu xây được surrogate PrLoS có cận hợp lệ, bổ sung SCA thành biến thể nâng cao. Mọi Taylor expansion chưa chắc là cận dưới.

**Repair:** hai thao tác ở Mục 5.1, áp lên các alert có slack thấp nhưng còn khả năng cứu; so sánh trên cùng tập seed thiết kế và cùng evaluator. Có giới hạn số lần thử mỗi vòng để đo runtime công bằng; B2 được cấp cùng ngân sách số lần evaluate để so sánh quality–runtime không thiên vị.

### 9.3. Trình tự một vòng

1. Khởi tạo quỹ đạo và relay/path khả thi.
2. Phân bổ bandwidth, đánh giá flow/deadline.
3. Thử cập nhật relay/path.
4. Thử cập nhật quỹ đạo và giải lại tài nguyên liên quan.
5. Chạy repair cho các cảnh báo có nguy cơ trễ.
6. Giữ ứng viên theo objective và tie-break cố định, ghi log.
7. Dừng theo số vòng, số lần không cải thiện hoặc time budget.

Một bước tìm kiếm thất bại không có nghĩa bài toán gốc vô nghiệm. Giữ nghiệm khả thi tốt nhất. Nghiệm không phục vụ cảnh báo nào là một kết quả chất lượng bằng 0, không tự coi là lỗi solver.

### 9.4. Phân tích và cận

Objective đơn điệu và bị chặn chỉ cho phép kết luận hội tụ giá trị, không chứng minh KKT hoặc tối ưu toàn cục. Không hứa hội tụ điểm dừng cho phương pháp local search/repair.

**Cận trên LP và gap.** Với quỹ đạo $q$ và candidate paths cố định, bài toán còn lại là MILP theo $(x, b, f, z)$. Nới lỏng $x, z\in[0,1]$; ràng buộc capacity $f\le\Delta t_e R(b)$ có $R$ lõm theo $b$ nên miền khả thi lồi; xấp xỉ $R$ bằng các tiếp tuyến (outer approximation) chỉ nới rộng miền, nên kết quả vẫn là cận trên hợp lệ:

$$
\mathrm{UB}(q)\ \ge\ \max_{x,b,f,z\ \text{nguyên}}\hat P_{\mathrm{timely}}(q).
$$

Báo trên mọi instance: $\mathrm{gap}=\big(\mathrm{UB}(q_P)-P\big)/\mathrm{UB}(q_P)$ với $q_P$ là quỹ đạo do P tìm được, và $\max\{\mathrm{UB}(q_{B1}),\mathrm{UB}(q_{B2}),\mathrm{UB}(q_P)\}$ làm cận rộng hơn. Cận này hợp lệ cho quỹ đạo và candidate set đã cho, không phải cận toàn cục của quỹ đạo liên tục; phải ghi rõ khi báo cáo. Với instance nhỏ, giải MILP chính xác để đối chiếu.

**Độ khó.** Ngay trên một liên kết dung lượng cố định, chọn tập alert giao đúng hạn với release time là bài lập lịch một máy tối thiểu số job trễ $1\,|\,r_j\,|\,\sum U_j$, đã được chứng minh NP-hard mạnh cho trường hợp không ngắt quãng; do đó bài toán đầy đủ NP-hard và heuristic là hợp lý. Khi cho phép chia bit theo slot (tương đương preemption), biến thể $1\,|\,r_j,\mathrm{pmtn}\,|\,\sum U_j$ có thuật toán đa thức trong tài liệu lập lịch và có thể dùng làm subroutine hoặc cận cho trường hợp một cổ chai. Cả hai kết quả phải được xác minh trên tài liệu gốc trước khi trích dẫn; không tuyên bố hardness cho biến thể có chia bit khi chưa có chứng minh.

Phân tích chi phí theo số alert $A$, relay $V$, cạnh $E$, slot $N$, candidate $K$, scenario $M$ và số vòng $I$. Với nhiều nhất $AK$ lần thử relay/vòng, mỗi lần tốn chi phí phân bổ/evaluate; cộng riêng số thử quỹ đạo và repair. Đếm số solver calls và runtime thực. Không dùng $O(n^3)$ chung mà thiếu mô tả solver.

## 10. Thực nghiệm và tiêu chí đánh giá

### 10.1. Thiết kế

- Cùng scenario, seed, thông tin kênh và ngân sách thời gian cho các phương pháp.
- 20–30 seed theo yêu cầu môn học; nêu seed điều khiển topology failure, target, workload, channel hay khởi tạo.
- Phân biệt số topology/ảnh độc lập với số lặp trên cùng scenario.
- Tập phát triển dùng chọn tham số; tập cuối và seed kênh đánh giá độc lập.
- Case study RescueNet có bản đồ target–relay–quỹ đạo và tiến trình giao cảnh báo.
- Nếu thiếu metadata địa lý, công bố giới hạn tính độc lập giữa các ảnh.

### 10.2. Chỉ số

Timely-alert ratio/probability là chỉ số chính. Báo thêm gap so với cận trên LP, $\mathrm{Conn}$, deadline-miss ratio, latency, dữ liệu chưa giao, runtime và số lần evaluate, sai số ràng buộc, scalability, robustness và sensitivity.

Với objective là biến Bernoulli, báo khoảng tin cậy và cách gộp theo scenario. Dùng so sánh ghép cặp giữa các phương pháp trên cùng scenario/seed, kiểm tra điều kiện kiểm định hoặc dùng permutation/Wilcoxon phù hợp. Báo effect size, không chỉ p-value; điều chỉnh khi có nhiều phép so sánh. Không coi mọi alert trong cùng một mạng là mẫu độc lập.

### 10.3. Ablation bắt buộc

- Bỏ repair nhưng giữ các khối khác (P → B2).
- Chỉ giữ thao tác chuyển băng thông; chỉ giữ thao tác đổi relay/path.
- Đổi objective sang max–min rate (P → B3) để tách tác dụng của objective khỏi tác dụng của cơ chế.
- Giữ relay cố định.
- Giữ quỹ đạo cố định.
- Chia đều bandwidth.
- Bỏ backlog khỏi tiêu chí repair.
- Dùng cùng compute budget hoặc báo đường quality–runtime để tránh quy lợi ích do chạy lâu hơn thành lợi ích cơ chế.

### 10.4. Sensitivity

Số alert/relay, kích thước cảnh báo, deadline slack, tỷ lệ hỏng mạng, tổng phổ, công suất cố định, thời gian nhiệm vụ, tốc độ/độ cao UAV, kênh PrLoS, số candidate paths, số scenario thiết kế, $\Delta$ và quy tắc chuyển đổi dataset.

Không quét toàn bộ tích tham số ngay từ đầu. Pilot xác định quy mô rồi khảo sát có kiểm soát. Với việc thêm vùng cấm trên cùng bài toán, tối ưu toàn cục không thể tăng do miền khả thi nhỏ hơn; cải thiện heuristic phải giải thích bằng khởi tạo/cực trị cục bộ.

## 11. Công cụ, cấu trúc và kiểm chứng

Python là lựa chọn mặc định. Dùng công cụ đọc graph, ảnh/mask, tối ưu lồi/MIP nếu cần và plotting. Chọn solver sau khi chạy thử khối nhỏ; MOSEK có thể dùng nếu có license phù hợp, không bắt buộc. GPU không cần cho phương pháp chính.

~~~text
data/raw/          dữ liệu gốc
data/manifests/    nguồn, checksum, phiên bản, license
data/processed/    scenario chuẩn
configs/           thông số mô hình và thí nghiệm
src/data/          parser/chuyển đổi
src/models/        kênh, graph theo thời gian, queue
src/baselines/     baseline
src/optimization/ phân rã và repair
experiments/       chạy và tổng hợp
tests/             kiểm chứng
results/           logs, metrics, figures
docs/              đặc tả, phương pháp, báo cáo
~~~

Kiểm chứng: zero bandwidth/power, slot đầu, release/deadline, buffer rỗng, backhaul bottleneck, graph đứt kết nối, không UAV, đoạn bay cắt vùng cấm, hai alert tranh cùng capacity, seed tái lập và dữ liệu giao không vượt nguồn. Evaluator độc lập với optimizer; dùng các instance nhỏ kiểm tra bằng tay hoặc solver.

Raw lớn không đưa vào lịch sử mã nguồn; công bố manifest và mã tạo scenario. Lưu solver version, môi trường, cấu hình, seed và hash scenario cùng mỗi run.

## 12. Tài liệu cần đọc

1. Tài liệu môn học, đề số 5 và yêu cầu hai pha.
2. Bài Huang 2025: [bản Markdown toàn văn](references/sensors-25-07443-v2.md) và DOI ở Mục 1.
3. Tài liệu chính thức SNDlib, Topology Zoo, RescueNet và bài mô tả dataset.
4. Boyd–Vandenberghe, [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/): concavity, relaxation, duality và constraints.
5. Từ [danh sách tài liệu tham khảo của bài Huang](references/sensors-25-07443-v2.md#references): Zeng–Wu–Zhang, “Accessing From the Sky” (tài liệu [24]); Li et al., “Full-Duplex UAV Relaying for Multiple User Pairs” ([13]); You–Zhang về PrLoS ([15]). Xác minh metadata và đọc toàn văn trước khi trích dẫn kết quả.
6. Tổng quan bổ sung: deadline-aware UAV relaying, relay selection với backhaul, time-expanded network flow, stochastic timely delivery, joint trajectory and bandwidth, decomposition và repair.

Lập bảng Related Work theo objective, số UAV, relay selection, deadline, connectivity, channel uncertainty, bandwidth/power, phương pháp và guarantee. Chốt baseline mạnh và tính mới sau tổng quan; chưa có bằng chứng đủ để tuyên bố novelty/SOTA.

## 13. Lộ trình theo khung 10 tuần

| Giai đoạn | Công việc | Điều kiện hoàn thành |
|---|---|---|
| Tuần 1–2 | v0 (Mục 3.4): pipeline Topology Zoo + demand SNDlib, evaluator và bộ test, B0/B1, cận trên LP | Bảng objective/gap/runtime trên ≥ 10 scenario, mã nguồn, feasibility |
| Tuần 3–6 | B2 (BCD kiểu Huang), B3, P với repair hai thao tác; benchmark đầy đủ; đánh giá trên realization | Ablation, sensitivity, 20–30 seed, kiểm định, gap |
| Tuần 7–9 | Baseline mạnh từ literature; SAA trong thiết kế nếu kịp; case study RescueNet; quality–runtime | So sánh literature, case study, gói tái lập |
| Tuần 10 | Hoàn thiện báo cáo | Abstract, Introduction, Related Work, Proposed Method, Experiment Results, Discussion, Conclusion |

Nếu thời gian hạn chế, giảm số mở rộng và quy mô trước; giữ đủ baseline, kiểm chứng và thực nghiệm theo yêu cầu môn học.

## 14. Tiêu chí nghiệm thu và giới hạn kết luận

- [ ] Đúng đề số 5: có trajectory, relay selection và bandwidth allocation.
- [ ] Objective đúng hạn được định nghĩa với release, deadline, số bit và xác suất.
- [ ] Connectivity được định nghĩa theo thời gian và đường đến trung tâm.
- [ ] Bài Huang được dùng đúng vai trò, khác biệt giao thức/mục tiêu được công khai.
- [ ] Dữ liệu có provenance, đơn vị và phép chuyển đổi tái lập.
- [ ] Baseline chạy trên benchmark công khai; thiếu nguồn nào được báo đúng.
- [ ] Có ít nhất một cơ chế cải tiến và ablation chứng minh tác động.
- [ ] Flow bảo toàn, buffer không âm, không dùng thông tin tương lai ngoài giả định.
- [ ] Tài nguyên tổng và quỹ đạo khả thi; không tăng công suất ngầm theo quy mô.
- [ ] Có 20–30 seed, kiểm định, effect size và uncertainty report.
- [ ] Có runtime, robustness, feasibility, scalability và sensitivity.
- [ ] Có cận trên LP và gap trên mọi instance; instance nhỏ đối chiếu với MILP chính xác.
- [ ] Connectivity được định nghĩa trên time-expanded graph và là mục tiêu phụ từ điển, dùng thống nhất cho mọi phương pháp.
- [ ] B2 kế thừa đúng cấu trúc BCD của bài Huang; B3 tách được tác dụng của objective.
- [ ] Có gói tái lập và báo cáo theo cấu trúc môn học.

Không đặt điều kiện phải tăng trước một tỷ lệ phần trăm nhất định. Kết quả không cải thiện vẫn phải báo cáo và phân tích. Mô phỏng không chứng minh hiệu năng thực địa; ảnh 2D không đo kênh; objective mẫu không tự bảo đảm xác suất ngoài mẫu; phương pháp heuristic không tự có bảo đảm tối ưu.

## 15. Trạng thái và bước tiếp theo

**Đã thống nhất:** phạm vi một UAV + relay mặt đất, mô phỏng, relay selection thực sự, deadline/connectivity theo đề số 5, phân rã và repair làm hướng phát triển; v0 cho Pha 1 (Mục 3.4); B2 kế thừa cấu trúc BCD của Huang và B3 dùng objective của Huang; cận trên LP và gap; connectivity là mục tiêu phụ từ điển; quy ước $L_{\mathrm{box}}=2$ km.

**Đề xuất cụ thể trong bản này:** hai pha cố định, công suất cố định, candidate paths hữu hạn, store-carry-forward, thiết kế trên tốc độ kỳ vọng ở Pha 1 và đánh giá ngoài mẫu trên realization. Những lựa chọn này cần được kiểm tra bằng pilot và điều chỉnh có ghi nhận nếu gặp bất nhất.

**Chưa thực hiện:** tải dataset; tổng quan xác nhận novelty và baseline mạnh; xác minh hai kết quả lập lịch ở Mục 9.4 trên tài liệu gốc; cài mô hình/thuật toán; chạy thực nghiệm. Bản này thay thế định hướng một UAV chỉ tối ưu scheduling/max–min rate ở bản nháp trước.

Bước tiếp theo là lập kế hoạch triển khai chi tiết cho v0, bắt đầu từ xác minh dataset và evaluator deadline/flow, sau khi rà soát đặc tả này.

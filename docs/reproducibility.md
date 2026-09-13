# Gói tái lập

Tài liệu này mô tả cách dựng lại môi trường, dữ liệu, kết quả, bảng và hình của báo cáo, và cách kiểm tra tự động rằng các file đã commit được sinh lại đúng. Thiết kế nằm ở spec D Mục 11–13 (`docs/superpowers/specs/2026-09-12-extensions-final-delivery-design.md`).

## 1. Môi trường

- Python 3.12 trong `.venv` do uv quản lý; `uv.lock` khoá phiên bản package. Tạo môi trường bằng `uv sync --extra test`.
- Thư viện chính: `numpy`, `scipy` (HiGHS cho quy hoạch tuyến tính), `cvxpy` với Clarabel (phương pháp TRAN), `networkx`, `pyproj`, Pillow, `PyYAML`.
- Báo cáo dùng pdfLaTeX, biber và script `compile_latex.sh` của `.claude/skills/latex-document-skill`.
- Kết quả đã commit được chạy trên WSL2 với 12 CPU và 20 GB RAM; môi trường cụ thể của từng thí nghiệm (Python, numpy, scipy, HiGHS, cvxpy, Clarabel, số CPU) nằm trong `manifest.json` của thí nghiệm đó.
- Các lệnh giải bài toán TRAN chạy trong `systemd-run --user --scope -p MemoryMax=... -p MemorySwapMax=0`, để một lỗi tràn bộ nhớ chỉ dừng đúng tiến trình đó.

## 2. Dữ liệu và giấy phép

Dữ liệu thô không nằm trong Git. Lệnh sau tải Topology Zoo (revision cố định), archive XML của SNDlib và archive validation của RescueNet (khoảng 2,37 GB), kiểm tra checksum và sinh dữ liệu đã xử lý:

```bash
uv run python -m data.cli all --profile configs/data/paper.yaml
uv run python -m data.cli verify --profile configs/data/paper.yaml
```

`data/manifests/sources.json` ghi URL, phiên bản, dung lượng và SHA-256 của từng file tải về. README của tác giả RescueNet ghi giấy phép CC BY-NC-ND cho nội dung dataset, nên kho này không phân phối lại ảnh, mask hay hình dựng từ chúng; kho chỉ lưu manifest lựa chọn cặp ảnh, manifest target và mã chuyển đổi.

## 3. Ba mức kiểm tra

`experiments/reproduce.py` ghi kết quả từng mức vào `results/reproducibility/<mức>.json` (danh sách kiểm tra đạt hoặc không đạt, commit, môi trường) và trả mã khác 0 khi có kiểm tra không đạt.

| Mức | Lệnh | Kiểm tra | Thời gian |
|---|---|---|---|
| `tables` | `uv run python experiments/reproduce.py --level tables` | Chạy lại thống kê và ba script sinh dữ liệu báo cáo vào thư mục tạm, so từng byte với `results/phase2/statistics*.csv` và `docs/report/data/` | vài giây |
| `scenarios` | `uv run python experiments/reproduce.py --level scenarios` | Kiểm tra checksum dữ liệu thô, sinh lại manifest target RescueNet và mọi họ scenario dưới một thư mục dữ liệu tạm, so với manifest đã commit và với SHA-256 ghi trong manifest kết quả | khoảng 20 giây |
| `experiments` | `systemd-run --user --scope --quiet -p MemoryMax=14G -p MemorySwapMax=0 uv run python experiments/reproduce.py --level experiments --subset Agis` | Chạy lại mọi cấu hình thí nghiệm trên các scenario của một topology, so từng dòng kết quả trên các cột không phải thời gian chạy | khoảng một giờ |

Bỏ `--subset` để chạy lại toàn bộ thí nghiệm và pilot TRAN; tổng thời gian tính đã ghi của từng thí nghiệm nằm trong `docs/report/data/ext_compute.csv` (bảng tính toán ở phụ lục báo cáo). Với `--subset`, bước đối chiếu MILP của Pha 1 được bỏ qua vì nó chọn instance trên toàn tập, còn pilot TRAN và `phase2_pilot` được bỏ qua vì chúng chạy trên tập phát triển, trong khi `--subset` nhận một topology đánh giá. `--experiments` giới hạn các cấu hình được chạy, và `--compare REFERENCE CANDIDATE` so hai thư mục kết quả bất kỳ theo cùng quy tắc.

Quy tắc so sánh dòng: `timely_count`, `connected_count` và `evaluate_calls` phải bằng nhau, `shortfall` lệch tối đa $10^{-9}$; giá trị cận LP lệch tương đối tối đa $10^{-6}$ và trạng thái phải bằng nhau; giá trị MILP chỉ được so khi cả hai lần chạy đều tối ưu; với pilot TRAN, số vòng lặp, trạng thái và tỷ lệ đúng hạn phải khớp. Chênh lệch trên máy khác được báo cáo, không bị lọc.

## 4. Nguồn gốc của kết quả

Mỗi script sinh dữ liệu báo cáo chỉ đọc một thí nghiệm khi manifest của nó hoàn tất và mô tả mã hiện tại (`runner.provenance.result_provenance_problem`): hoặc `source_hash` bằng mã hiện tại, hoặc kết quả đến từ một commit sạch là tổ tiên của HEAD và không file nào của `src/models`, `src/baselines`, `src/bounds`, `src/optimization`, `src/literature` bị sửa, xoá hay đổi tên kể từ commit đó. Package `runner` không nằm trong danh sách vì chỉ điều phối task; mức `experiments` kiểm tra nó từ đầu đến cuối. Kết quả `results/phase2/pilot/` của sub-project C là bản ghi lịch sử của lần chạy thử và không được báo cáo dùng.

## 5. Bảng, hình và nguồn dữ liệu

| Mục báo cáo | Dữ liệu | Script sinh | Thí nghiệm |
|---|---|---|---|
| Bảng tham số (`tab:params`), bảng MILP (`tab:phase1-milp`), hình thời gian LP (`fig:lp-runtime`) | `params.csv`, `phase1_milp.csv`, `lp_runtime.csv`, `phase1_results.tex` | `experiments/report_tables.py` | `configs/experiments/phase1.yaml` → `results/phase1/` |
| So sánh chính, thống kê, ablation, kết nối, số realization thiết kế (`tab:phase2-*`), ngân sách (`fig:budget`), độ nhạy (`fig:sensitivity`) | `phase2_*.csv`, `phase2_results.tex` | `experiments/report_tables_phase2.py` (thống kê: `experiments/phase2_statistics.py`) | `phase2_main`, `phase2_budget`, `phase2_sensitivity`, `phase2_design` → `results/phase2/` |
| Bảng tổng quan tài liệu (`tab:related`) | `related_work.csv` | viết tay từ `docs/literature/review.md` | --- |
| So sánh với TRAN (`tab:literature`, `tab:literature-stats`), chất lượng–thời gian (`fig:quality-runtime`) | `ext_literature.csv`, `ext_literature_statistics.csv`, `ext_quality_points.csv`, `phase2_budget_*.csv` | `experiments/report_tables_extensions.py` (thống kê: `experiments/phase2_literature_statistics.py`) | `experiments/tran_pilot.py` → `results/phase2/tran_pilot/`; `phase2_literature` → `results/phase2/literature/` |
| Case study (`tab:case-study`), bản đồ (`fig:case-map`), tiến trình giao cảnh báo (`fig:case-progress`) | `ext_case_study.csv`, `ext_map_*.csv`, `ext_progress.csv` | `experiments/report_tables_extensions.py` | `data.cli rescuenet-targets`, `data.cli scenarios --config configs/scenarios/rescuenet.yaml`, `phase2_case_study` → `results/phase2/case_study/`, `experiments/case_study_trace.py` → `results/phase2/case_study/trace/` |
| Thời gian tính (`tab:compute`) | `ext_compute.csv`, `ext_results.tex` | `experiments/report_tables_extensions.py` | mọi thí nghiệm ở trên |

## 6. Đối chiếu tiêu chí nghiệm thu

| Tiêu chí (`docs/issue.md` Mục 14) | Bằng chứng |
|---|---|
| Đúng đề số 5: có trajectory, relay selection và bandwidth allocation | Báo cáo Mục Mô hình và phát biểu bài toán; `src/models/plan.py` (quỹ đạo, lựa chọn đường, băng thông) |
| Objective đúng hạn được định nghĩa với release, deadline, số bit và xác suất | Báo cáo công thức mục tiêu; `src/models/metrics.py`, `src/models/evaluate.py` |
| Connectivity được định nghĩa theo thời gian và đường đến trung tâm | Báo cáo Mục phát biểu bài toán; `src/models/metrics.py` |
| Bài Huang được dùng đúng vai trò, khác biệt giao thức/mục tiêu được công khai | Báo cáo Mục Nghiên cứu liên quan và Mục phương pháp tìm kiếm (vì sao các khối không lồi, B3) |
| Dữ liệu có provenance, đơn vị và phép chuyển đổi tái lập | `data/manifests/`, `configs/data/`, `experiments/reproduce.py --level scenarios` |
| Baseline chạy trên benchmark công khai; thiếu nguồn nào được báo đúng | B0, B1 và TRAN trên kịch bản từ Topology Zoo và SNDlib (`tab:phase2-main`, `tab:literature`); giả định tiêu cự của ảnh FC2103 và thiếu GSD trong bài RescueNet được ghi ở spec D Mục 17 |
| Có ít nhất một cơ chế cải tiến và ablation chứng minh tác động | P và bảy biến thể ablation (`tab:phase2-ablation`), kết quả âm tính được báo cáo |
| Flow bảo toàn, buffer không âm, không dùng thông tin tương lai ngoài giả định | Bộ kiểm tra độc lập `src/models/checker.py` chạy ở mọi lần đánh giá cuối; `tests/models/` |
| Tài nguyên tổng và quỹ đạo khả thi; không tăng công suất ngầm theo quy mô | `validate_plan` trong `src/models/plan.py`; công suất cố định cho mọi phương pháp, kể cả TRAN |
| Có 20–30 seed, kiểm định, effect size và uncertainty report | 30 realization cho mỗi scenario; `tab:phase2-stats`, `tab:literature-stats` (kiểm định đổi dấu chính xác, Holm, bootstrap, rank-biserial) |
| Có runtime, robustness, feasibility, scalability và sensitivity | Mục thời gian tính, `fig:sensitivity`, `fig:quality-runtime`, `tab:case-study`, `tab:compute` |
| Có cận trên LP và gap trên mọi instance; instance nhỏ đối chiếu với MILP chính xác | `bounds.csv` của mọi thí nghiệm, cột gap của các bảng kết quả, `tab:phase1-milp` |
| Connectivity trên time-expanded graph là mục tiêu phụ từ điển, dùng thống nhất | Khoá từ điển trong `src/models/evaluate.py` dùng cho mọi phương pháp |
| B2 kế thừa đúng cấu trúc BCD của bài Huang; B3 tách được tác dụng của objective | Báo cáo Mục phương pháp tìm kiếm; so sánh B2 $-$ B3 trong `tab:phase2-stats` |
| Có gói tái lập và báo cáo theo cấu trúc môn học | Tài liệu này, `experiments/reproduce.py`, `results/reproducibility/*.json`; báo cáo có Tóm tắt, Giới thiệu, Nghiên cứu liên quan, Phương pháp, Kết quả, Thảo luận, Kết luận |

## 7. Chạy lại toàn bộ theo thứ tự

```bash
uv sync --extra test
uv run python -m data.cli all --profile configs/data/paper.yaml
uv run python -m data.cli rescuenet-targets --config configs/data/rescuenet_targets.yaml
for config in configs/scenarios/*.yaml; do uv run python -m data.cli scenarios --config "$config"; done
uv run python experiments/run_phase1.py --config configs/experiments/phase1.yaml
for name in phase2_main phase2_budget phase2_sensitivity phase2_design; do uv run python experiments/run_phase2.py --config configs/experiments/$name.yaml; done
uv run python experiments/phase2_statistics.py
systemd-run --user --scope --quiet -p MemoryMax=14G -p MemorySwapMax=0 uv run python experiments/tran_pilot.py
systemd-run --user --scope --quiet -p MemoryMax=14G -p MemorySwapMax=0 uv run python experiments/run_phase2.py --config configs/experiments/phase2_literature.yaml
uv run python experiments/phase2_literature_statistics.py
systemd-run --user --scope --quiet -p MemoryMax=14G -p MemorySwapMax=0 uv run python experiments/run_phase2.py --config configs/experiments/phase2_case_study.yaml
systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 uv run python experiments/case_study_trace.py
uv run python experiments/report_tables.py
uv run python experiments/report_tables_phase2.py
uv run python experiments/report_tables_extensions.py
```

Sau đó biên dịch báo cáo bằng script của skill LaTeX như trong README. Nếu editor tự biên dịch `docs/report` khi file `.tex` thay đổi, hãy tắt tính năng đó hoặc biên dịch trong một bản sao ngoài workspace, vì hai lần biên dịch đồng thời ghi đè cùng `main.pdf`.

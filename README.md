# hust-algorithms-design-analysis

## Chuẩn bị dữ liệu nghiên cứu

Pipeline dùng Python 3.11 trở lên và lưu dữ liệu tải về bên ngoài Git. Môi trường được quản lý bằng [uv](https://docs.astral.sh/uv/) trong `.venv` của project; `uv.lock` ghi phiên bản đã khóa. Tạo môi trường và cài dependencies bằng:

```bash
uv sync --extra test
```

Chạy kiểm tra cấu hình mà không tải file:

```bash
uv run python -m data.cli all --profile configs/data/paper.yaml --dry-run
```

Chạy profile paper để tải Topology Zoo, archive XML SNDlib và archive validation RescueNet khoảng 2.37 GB, sau đó xác minh và tạo dữ liệu xử lý:

```bash
uv run python -m data.cli all --profile configs/data/paper.yaml
uv run python -m data.cli verify --profile configs/data/paper.yaml
```

Profile `pilot` dùng để kiểm tra pipeline với descriptor RescueNet nhưng không tải archive ảnh. Profile `full-rescuenet` thêm archive train và test, tổng cộng khoảng 21 GB. Raw data và dữ liệu processed lớn nằm trong các thư mục bị Git ignore. Nếu checksum, schema hoặc license evidence không hợp lệ, pipeline dừng và giữ lại thông tin lỗi; không tự chuyển sang mirror.

Chạy test offline bằng:

```bash
uv run --extra test pytest -q
```

## Bộ scenario v0

Sau khi chạy profile `paper`, sinh bộ scenario v0 gồm 30 scenario đánh giá và 6 scenario phát triển:

```bash
uv run python -m data.cli scenarios --config configs/scenarios/v0.yaml
```

Lệnh đọc network đã chuẩn hóa trong `data/processed/networks/` và instance SNDlib `abilene`, ghi file scenario vào `data/processed/scenarios/v0/` (bị Git ignore) và manifest `data/manifests/scenarios_v0.csv` (được commit). Chạy lại với cùng cấu hình cho cùng cột `sha256`.

## Evaluator và kiểm tra hiệu chỉnh

Mọi phương pháp chấm điểm kế hoạch qua cùng một hàm (chạy trong môi trường bằng `uv run python`):

```python
from pathlib import Path

from models.evaluate import REALIZED, evaluate
from models.paths import candidate_paths
from models.plan import EqualSplitBacklogged, Plan, stationary_trajectory
from models.scenario import load_scenario

scenario = load_scenario(Path("data/processed/scenarios/v0/Agis-r0.json"))
candidates = candidate_paths(scenario)
plan = Plan(stationary_trajectory(scenario), {alert.id: 0 for alert in scenario.alerts}, EqualSplitBacklogged())
result = evaluate(scenario, plan, REALIZED, tuple(range(30)), candidates=candidates)
print(result.timely_ratio, result.key)
```

Kiểm tra hiệu chỉnh chạy hai kế hoạch đơn giản trên 30 scenario đánh giá với 30 realization kênh và ghi `results/calibration/v0.json`:

```bash
uv run python experiments/calibrate_v0.py
```

Lệnh trả mã 0 khi cả hai kế hoạch khả thi và tỷ lệ đúng hạn trung bình của mỗi kế hoạch nằm trong [0.05, 0.95].

## Pha 1: baseline, cận trên và báo cáo

Sinh họ scenario backhaul yếu `v0-bh50` (giống v0, chỉ đổi dung lượng backhaul còn 50 kbit/s) và kiểm tra hiệu chỉnh:

```bash
uv run python -m data.cli scenarios --config configs/scenarios/v0-bh50.yaml
uv run python experiments/calibrate_v0.py --manifest data/manifests/scenarios_v0-bh50.csv --scenario-dir data/processed/scenarios/v0-bh50 --output results/calibration/v0-bh50.json
```

Chạy thí nghiệm Pha 1 (B0, B1, cận trên LP theo từng realization và đối chiếu MILP) trên cả hai họ scenario:

```bash
uv run python experiments/run_phase1.py --config configs/experiments/phase1.yaml
```

Kết quả nằm trong `results/phase1/`; `results/phase1/shards/` (bị Git ignore) lưu từng task nên chạy lại lệnh sẽ bỏ qua task đã xong. Lệnh trả mã khác 0 nếu có task lỗi, LP không tối ưu hoặc kiểm tra tính hợp lệ của cận thất bại.

Sinh dữ liệu cho báo cáo (bảng tham số, bảng kết quả, macro số liệu, dữ liệu hình runtime) rồi biên dịch báo cáo bằng script của skill LaTeX:

```bash
uv run python experiments/report_tables.py
cd docs/report && bash ../../.claude/skills/latex-document-skill/scripts/compile_latex.sh main.tex --preview --preview-dir build/preview
```

## Pha 2: phương pháp tìm kiếm, thí nghiệm và thống kê

Sinh 13 họ scenario độ nhạy; mỗi họ chỉ khác `configs/scenarios/v0.yaml` ở các trường liệt kê trong spec Pha 2, Mục 11.3:

```bash
for config in configs/scenarios/sens-*.yaml; do uv run python -m data.cli scenarios --config "$config"; done
```

Chạy các thí nghiệm Pha 2. Kết quả nằm trong `results/phase2/<tên>/`; thư mục `shards/` bên trong bị Git ignore và giúp chạy lại bỏ qua task đã xong:

```bash
uv run python experiments/run_phase2.py --config configs/experiments/phase2_pilot.yaml        # chạy thử B1, B2, B3, P trên tập phát triển
uv run python experiments/run_phase2.py --config configs/experiments/phase2_main.yaml         # B0, B1, B2, B3, P và bảy biến thể ablation trên v0, v0-bh50
uv run python experiments/run_phase2.py --config configs/experiments/phase2_budget.yaml       # B2 và P với ngân sách 250 đến 4000 lần đánh giá
uv run python experiments/run_phase2.py --config configs/experiments/phase2_sensitivity.yaml  # B1, B2, P trên replicate 0 của 15 họ scenario
uv run python experiments/run_phase2.py --config configs/experiments/phase2_design.yaml       # P thiết kế trên tốc độ kỳ vọng, 1 hoặc 10 realization
```

Tính kiểm định ghép cặp theo topology (tám so sánh, hiệu chỉnh Holm) từ thí nghiệm chính:

```bash
uv run python experiments/phase2_statistics.py
```

Tổng quan tài liệu, bảng so sánh và lý do chọn baseline từ tài liệu nằm ở `docs/literature/review.md`.

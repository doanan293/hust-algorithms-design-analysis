# hust-algorithms-design-analysis

## Chuẩn bị dữ liệu nghiên cứu

Pipeline dùng Python 3.11 trở lên và lưu dữ liệu tải về bên ngoài Git. Cài dependencies bằng:

```bash
python -m pip install --user --break-system-packages -e '.[test]'
```

Chạy kiểm tra cấu hình mà không tải file:

```bash
python -m data.cli all --profile configs/data/paper.yaml --dry-run
```

Chạy profile paper để tải Topology Zoo, archive XML SNDlib và archive validation RescueNet khoảng 2.37 GB, sau đó xác minh và tạo dữ liệu xử lý:

```bash
python -m data.cli all --profile configs/data/paper.yaml
python -m data.cli verify --profile configs/data/paper.yaml
```

Profile `pilot` dùng để kiểm tra pipeline với descriptor RescueNet nhưng không tải archive ảnh. Profile `full-rescuenet` thêm archive train và test, tổng cộng khoảng 21 GB. Raw data và dữ liệu processed lớn nằm trong các thư mục bị Git ignore. Nếu checksum, schema hoặc license evidence không hợp lệ, pipeline dừng và giữ lại thông tin lỗi; không tự chuyển sang mirror.

Chạy test offline bằng:

```bash
python -m pytest -q
```

## Bộ scenario v0

Sau khi chạy profile `paper`, sinh bộ scenario v0 gồm 30 scenario đánh giá và 6 scenario phát triển:

```bash
python -m data.cli scenarios --config configs/scenarios/v0.yaml
```

Lệnh đọc network đã chuẩn hóa trong `data/processed/networks/` và instance SNDlib `abilene`, ghi file scenario vào `data/processed/scenarios/v0/` (bị Git ignore) và manifest `data/manifests/scenarios_v0.csv` (được commit). Chạy lại với cùng cấu hình cho cùng cột `sha256`.

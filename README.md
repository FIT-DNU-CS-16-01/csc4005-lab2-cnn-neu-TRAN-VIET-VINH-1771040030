# Lab2: CNN Image Classification (From Scratch vs Transfer Learning)

## 📝 Thông tin sinh viên

- **Họ tên:** Trần Việt Vinh
- **MSSV:** 1771040030
- **Môn học:** HOC SAU

---

## 📁 Cấu trúc repo

```
csc4005-lab2-cnn-neu-TRAN-VIET-VINH-1771040030/
├── README.md
├── requirements.txt
├── REPORT_LAB2.md
├── configs/
│   ├── baseline_scratch.json
│   └── baseline_transfer.json
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
├── data/
│   └── NEU-CLS.zip
├── outputs/
│   ├── cnn_small_baseline_offline/
│   │   ├── best_model.pt
│   │   ├── history.csv
│   │   ├── curves.png
│   │   ├── confusion_matrix.png
│   │   └── metrics.json
│   └── resnet18_transfer_offline/
│       ├── best_model.pt
│       ├── history.csv
│       ├── curves.png
│       ├── confusion_matrix.png
│       └── metrics.json
└── wandb/
    ├── offline-run-20260416_151714-tn5l6ir0/
    └── offline-run-20260416_154350-jkm1cwd7/
```

---

## 🛠️ Cài đặt môi trường

**1. Tạo môi trường Conda (nếu chưa có)**

```bash
conda create -n csc4005-dl python=3.10 -y
conda activate csc4005-dl
```

**2. Cài thư viện**

```bash
pip install -r requirements.txt
```

**3. Đăng nhập W&B (tùy chọn, nếu muốn log lên cloud)**

```bash
wandb login
# Dán API key từ https://wandb.ai/authorize
```

---

## 📂 Chuẩn bị dữ liệu

1. Tải file `NEU-CLS.zip` từ [NEU Surface Defect Database](http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html)
2. Tạo thư mục `data/` trong repo
3. Đặt file `NEU-CLS.zip` vào thư mục `data/`

```
data/
└── NEU-CLS.zip
```

Kiểm tra dữ liệu:

```bash
python -c "import zipfile; z = zipfile.ZipFile('./data/NEU-CLS.zip'); print('\n'.join(z.namelist()[:10]))"
```

Kết quả mong đợi:

```
train/
train/train/
train/train/images/
train/train/images/crazing_10.jpg
...
```

---

## 🚀 Chạy huấn luyện

### 🔹 CNN from scratch

```bash
python -m src.train \
  --data_dir ./data/NEU-CLS.zip \
  --run_name cnn_small_baseline \
  --model_name cnn_small \
  --train_mode scratch \
  --optimizer adamw \
  --lr 0.001 \
  --weight_decay 0.0001 \
  --dropout 0.3 \
  --epochs 20 \
  --batch_size 32 \
  --img_size 64 \
  --patience 5 \
  --augment
```

### 🔹 Transfer Learning (ResNet18)

```bash
python -m src.train \
  --data_dir ./data/NEU-CLS.zip \
  --run_name resnet18_transfer \
  --model_name resnet18 \
  --train_mode transfer \
  --optimizer adamw \
  --lr 0.001 \
  --weight_decay 0.0001 \
  --dropout 0.3 \
  --epochs 10 \
  --batch_size 32 \
  --img_size 128 \
  --patience 3 \
  --augment
```

### 🔹 Fine-tune ResNet18 (tùy chọn)

```bash
python -m src.train \
  --data_dir ./data/NEU-CLS.zip \
  --run_name resnet18_finetune \
  --model_name resnet18 \
  --train_mode finetune \
  --optimizer adamw \
  --lr 0.0001 \
  --weight_decay 0.0001 \
  --dropout 0.3 \
  --epochs 10 \
  --batch_size 32 \
  --img_size 128 \
  --patience 3 \
  --augment
```

---

## 📊 Kết quả đạt được

### CNN from scratch

| Chỉ số | Giá trị |
|--------|---------|
| Best validation accuracy | 94.44% |
| Test accuracy | 94.81% |
| Thời gian trung bình/epoch | 3.45 giây |
| Số lượng tham số | 32,614 |
| Epoch đạt best | 14, 16, 18 |

### Transfer Learning (ResNet18)

| Chỉ số | Giá trị |
|--------|---------|
| Best validation accuracy | 96.67% |
| Test accuracy | 96.30% |
| Thời gian trung bình/epoch | 19.23 giây |
| Số lượng tham số | 11,179,590 |
| Epoch đạt best | 10 |

---

## 📈 Learning curves

Sau khi chạy, learning curves được tự động lưu tại:

```
outputs/<run_name>/curves.png
```

- CNN scratch: `outputs/cnn_small_baseline_offline/curves.png`
- ResNet18 Transfer: `outputs/resnet18_transfer_offline/curves.png`

---

## 🔗 W&B Dashboard

Sau khi sync, có thể xem kết quả online tại:

| Mô hình | Link |
|---------|------|
| CNN from scratch | [tn5l6ir0](https://wandb.ai/vinhviettran955-no/csc4005-lab2-neu-cnn/runs/tn5l6ir0) |
| ResNet18 Transfer | [jkm1cwd7](https://wandb.ai/vinhviettran955-no/csc4005-lab2-neu-cnn/runs/jkm1cwd7) |

> **Lưu ý:** Cần đăng nhập bằng tài khoản `vinhviettran955-no` để xem dashboard.

---

## 📁 Cấu trúc thư mục output

Sau khi chạy xong, mỗi run sẽ tạo một thư mục trong `outputs/`:

```
outputs/
├── cnn_small_baseline_offline/
│   ├── best_model.pt          # Model tốt nhất (theo val acc)
│   ├── history.csv            # Lịch sử train/val theo từng epoch
│   ├── curves.png             # Learning curves (loss & accuracy)
│   ├── confusion_matrix.png   # Ma trận nhầm lẫn trên test set
│   └── metrics.json           # Tổng hợp các metrics chính
└── resnet18_transfer_offline/
    ├── best_model.pt
    ├── history.csv
    ├── curves.png
    ├── confusion_matrix.png
    └── metrics.json
```
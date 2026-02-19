# Linly-Dubbing: Công cụ Lồng tiếng & Dịch thuật Video AI chuyên nghiệp (Studio-Grade)

Linly-Dubbing là một giải pháp lồng tiếng video đa ngôn ngữ toàn diện, tập trung vào chất lượng âm thanh chuyên nghiệp và khả năng cá nhân hóa cao cho người Việt.

## 🌟 Tính năng nổi bật

- **Kiến trúc NestJS-Style**: Mã nguồn được tổ chức chuyên nghiệp trong thư mục `src/`.
- **Lồng tiếng Một Chạm (One-Touch Dubbing)**: Xử lý toàn bộ quy trình từ tách âm thanh, dịch thuật đến ghép video chỉ với một nút bấm.
- **Studio-Grade Audio**:
  - **Dereverb**: Khử vang môi trường cho giọng nói.
  - **Broadcast Mastering**: Tối ưu hóa chất lượng âm thanh phòng thu.
  - **Sidechain Ducking**: Tự động giảm âm lượng nhạc nền khi có tiếng nói.
- **Voice Clone Cao cấp**: Hỗ trợ XTTS, CosyVoice và mô hình VITS (VoxCPM) tối ưu cho tiếng Việt.
- **Việt hóa 100%**: Giao diện và thông báo hệ thống hoàn toàn bằng tiếng Việt.

## 🚀 Hướng dẫn khởi chạy

Cấu trúc mới đã chuyển toàn bộ mã nguồn vào thư mục `src/`. Bạn có thể chạy các thành phần bằng lệnh sau:

### 1. Giao diện Desktop (PySide6)
Sử dụng đầy đủ các tính năng cấu hình và xem trước video:
```bash
python src/main.py
```

### 2. Giao diện Web (Gradio)
Tiện lợi để chạy trên máy chủ hoặc qua trình duyệt:
```bash
python src/web.py
```

### 3. Giao diện Dòng lệnh (CLI)
Dành cho xử lý hàng loạt hoặc tự động hóa:
```bash
python src/cli.py /đường/dẫn/video.mp4
```

### 4. Công cụ so sánh Voice Clone
So sánh chất lượng giữa XTTS, CosyVoice và VoxCPM:
```bash
python compare_cloning.py /đường/dẫn/video_mau.mp4
```

## 📂 Cấu trúc thư mục đầu ra

Mọi kết quả xử lý sẽ được lưu tập trung tại thư mục:
- **`outputs/`**: Chứa các thư mục dự án được đặt tên theo tên video gốc.
- **`outputs/comparisons/`**: Lưu kết quả từ công cụ so sánh voice clone.

## 🛠️ Cấu hình

Bạn có thể chỉnh sửa cấu hình mặc định (ngôn ngữ, thiết bị sử dụng, API keys...) trong tab **Cấu hình hệ thống** trên giao diện Desktop hoặc chỉnh sửa trực tiếp tệp `src/ui/pyside/tabs/config.json`.

---
*Phát triển và tối ưu hóa bởi Đội ngũ Linly-Dubbing Vietnamese.*

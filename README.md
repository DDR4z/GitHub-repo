# AI Clip Maker — Streamlit Cloud (Auto Images + TTS + FFmpeg)

สร้างวิดีโอจากสคริปต์ **ออนไลน์ 100%**:
- gTTS สร้างเสียงพูด (ไทย/อังกฤษ)
- สร้างภาพประกอบอัตโนมัติ (เลือก Pexels API หรือ HuggingFace Inference API)
- รวมภาพ+เสียงเป็น MP4 ด้วย FFmpeg (ผ่าน `imageio-ffmpeg`)

## ใช้งานบน Streamlit Cloud
1) สร้าง GitHub repo ว่าง ๆ แล้วอัปไฟล์ต่อไปนี้ขึ้นไป:
   - `app.py`
   - `requirements.txt`
   - `README.md`
2) ไปที่ https://share.streamlit.io/ → Deploy app → ชี้ไปที่ repo + สาขา + ไฟล์ `app.py`
3) ตั้งค่า **Secrets** ใน Streamlit Cloud (ถ้ามีคีย์):
   ```toml
   # ในหน้า App → Settings → Secrets
   PEXELS_API_KEY = "xxxxx"
   HUGGING_FACE_HUB_TOKEN = "hf_xxx"
   HF_MODEL_ID = "stabilityai/stable-diffusion-2-1"
   ```
4) เปิดแอป → กรอกสคริปต์ → เลือกโหมดรูป → กด 🚀 สร้างคลิป → กด ⬇️ ดาวน์โหลด MP4

## รันโลคัล (ทางเลือก)
```bash
pip install -r requirements.txt
streamlit run app.py
```

> หมายเหตุ
- HuggingFace Inference API บางโมเดลอาจต้องอยู่ในคิว/ใช้เวลานาน หากไม่รีบ แนะนำ Pexels หรืออัปโหลดรูปเอง
- ถ้าเบราว์เซอร์ไม่ดาวน์โหลดอัตโนมัติ ใช้ปุ่ม "⬇️ ดาวน์โหลด MP4" ที่หน้าแอป

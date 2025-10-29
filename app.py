import os, io, math, re, time, tempfile, subprocess, json
from pathlib import Path
import streamlit as st
from gtts import gTTS
from PIL import Image
import requests
import imageio_ffmpeg

st.set_page_config(page_title="AI Clip Maker", page_icon="🎬", layout="centered")

st.title("🎬 AI Clip Maker — Online")
st.caption("ใส่สคริปต์ → ให้ระบบสร้างภาพอัตโนมัติ (Pexels หรือ HuggingFace Inference API) → สร้างเสียงด้วย gTTS → รวมเป็น MP4 ด้วย FFmpeg → ดาวน์โหลด")

# ---------- Helpers ----------
def split_script_to_prompts(text, n):
    s = re.split(r'[\n\r\.!?]+', text)
    s = [t.strip() for t in s if t.strip()]
    if not s:
        s = [text.strip()]
    out = []
    if len(s) <= n:
        out = (s + [s[-1]] * n)[:n]
    else:
        size = math.ceil(len(s)/n)
        for i in range(0, len(s), size):
            chunk = ' '.join(s[i:i+size])
            out.append(chunk)
        out = out[:n]
    return out

def ensure_dirs(tmpdir):
    (tmpdir / "input_images").mkdir(parents=True, exist_ok=True)
    (tmpdir / "output").mkdir(parents=True, exist_ok=True)
    (tmpdir / "tmpframes").mkdir(parents=True, exist_ok=True)

def pexels_fetch_images(api_key, prompts, save_dir):
    saved = []
    headers = {"Authorization": api_key}
    for i, prompt in enumerate(prompts, 1):
        q = prompt[:60] if prompt else "abstract background"
        params = {"query": q, "per_page": 10, "orientation": "landscape"}
        r = requests.get("https://api.pexels.com/v1/search", headers=headers, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if not data.get("photos"):
            continue
        photo = data["photos"][0]
        url = photo["src"].get("large2x") or photo["src"].get("large") or photo["src"].get("original")
        if not url:
            continue
        img = requests.get(url, timeout=30)
        img.raise_for_status()
        path = save_dir / f"auto_{i:03d}.jpg"
        with open(path, "wb") as f:
            f.write(img.content)
        saved.append(path)
    return saved

def hf_inference_image(token, model_id, prompt):
    # HuggingFace Inference API - returns bytes (png/jpeg) for image
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.post(url, headers=headers, json={"inputs": prompt}, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"Inference API error {resp.status_code}: {resp.text[:200]}")
    return resp.content

def hf_generate_images(token, model_id, prompts, save_dir):
    saved = []
    for i, p in enumerate(prompts, 1):
        img_bytes = hf_inference_image(token, model_id, p)
        path = save_dir / f"auto_{i:03d}.png"
        with open(path, "wb") as f:
            f.write(img_bytes)
        saved.append(path)
    return saved

def build_video(images, audio_path, out_path, seconds_per_image):
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    tmpdir = out_path.parent / "tmpframes"
    tmpdir.mkdir(parents=True, exist_ok=True)

    # Prepare frames and concat list
    list_path = tmpdir / "list.txt"
    if list_path.exists():
        list_path.unlink()

    # Convert each image to 1920x1080 cover
    for idx, img_path in enumerate(images, 1):
        out_img = tmpdir / f"img{idx:03d}.jpg"
        cmd = [
            ffmpeg_path, "-y", "-i", str(img_path),
            "-vf", "scale=1920:1080:force_original_aspect_ratio=cover,crop=1920:1080",
            str(out_img)
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with open(list_path, "a", encoding="utf-8") as f:
            f.write(f"file '{out_img.as_posix()}'\n")
            f.write(f"duration {seconds_per_image}\n")

    # Make silent video
    video_noaudio = tmpdir / "video_noaudio.mp4"
    cmd = [ffmpeg_path, "-y", "-f", "concat", "-safe", "0", "-i", str(list_path),
           "-r", "30", "-pix_fmt", "yuv420p", str(video_noaudio)]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Merge audio
    cmd = [ffmpeg_path, "-y", "-i", str(video_noaudio), "-i", str(audio_path),
           "-shortest", "-c:v", "libx264", "-c:a", "aac", "-b:a", "192k", str(out_path)]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# ---------- UI ----------
st.subheader("ตั้งค่าโปรเจกต์")
default_script = "วันนี้เราจะให้ AI ช่วยคิดภาพประกอบตามสคริปต์ แล้วรวมเป็นวิดีโออัตโนมัติ"
script = st.text_area("สคริปต์", value=default_script, height=150)
lang = st.selectbox("ภาษาเสียงพูด (gTTS)", ["th", "en"], index=0, help="ไทยใช้ th, อังกฤษใช้ en")
seconds = st.slider("วินาทีต่อภาพ", min_value=2, max_value=12, value=4, step=1)
scenes = st.slider("จำนวนภาพ (ฉาก)", min_value=1, max_value=12, value=4, step=1)
filename = st.text_input("ชื่อไฟล์เอาต์พุต", value="ai-clip.mp4")

st.markdown("---")
st.subheader("โหมดสร้างภาพอัตโนมัติ (เลือกอย่างใดอย่างหนึ่ง หรืออัปโหลดเอง)")
mode = st.radio("เลือกโหมด", ["อัปโหลดภาพเอง", "Pexels API", "HuggingFace Inference API"], index=0)

# Secrets
pexels_key = st.text_input("PEXELS_API_KEY", value=st.secrets.get("PEXELS_API_KEY", ""), type="password")
hf_token = st.text_input("HUGGING_FACE_HUB_TOKEN", value=st.secrets.get("HUGGING_FACE_HUB_TOKEN", ""), type="password")
hf_model = st.text_input("HF Model (Inference API)", value=st.secrets.get("HF_MODEL_ID", "stabilityai/stable-diffusion-2-1"))

uploaded = st.file_uploader("อัปโหลดภาพ (เลือกหลายไฟล์)", type=["jpg","jpeg","png"], accept_multiple_files=True)

go = st.button("🚀 สร้างคลิป")

if go:
    with st.spinner("กำลังสร้างคลิป..."):
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            ensure_dirs(tmpdir)
            input_dir = tmpdir / "input_images"
            output_dir = tmpdir / "output"
            output_dir.mkdir(exist_ok=True, parents=True)

            # 1) Prepare images
            prompts = split_script_to_prompts(script, scenes)
            saved_paths = []
            if mode == "อัปโหลดภาพเอง" and uploaded:
                for f in uploaded:
                    p = input_dir / f.name
                    p.write_bytes(f.read())
                    saved_paths.append(p)
            elif mode == "Pexels API" and pexels_key:
                saved_paths = pexels_fetch_images(pexels_key, prompts, input_dir)
            elif mode == "HuggingFace Inference API" and hf_token and hf_model:
                try:
                    saved_paths = hf_generate_images(hf_token, hf_model, prompts, input_dir)
                except Exception as e:
                    st.error(f"HuggingFace Inference ล้มเหลว: {e}")
                    saved_paths = []

            # 2) TTS
            audio_mp3 = output_dir / "audio.mp3"
            try:
                t=gTTS(text=script, lang=lang)
                t.save(str(audio_mp3))
            except Exception as e:
                st.error(f"สร้างเสียงล้มเหลว: {e}")
                st.stop()

            # 3) Build video
            out_mp4 = output_dir / filename
            try:
                if saved_paths:
                    build_video(saved_paths, audio_mp3, out_mp4, seconds)
                else:
                    # Black background fallback (length of audio)
                    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
                    # Get audio duration
                    # Use ffprobe packaged with imageio-ffmpeg (same path)
                    # Sometimes ffprobe is not bundled; fallback to a fixed duration
                    try:
                        # probe via ffmpeg -i and parse duration is messy; simpler: mux black video longer than audio and -shortest
                        pass
                    except:
                        pass
                    # Make 30-second black if no images (approx); then shortest merges
                    black = tmpdir / "black.mp4"
                    subprocess.run([ffmpeg_path, "-y", "-f", "lavfi", "-i", "color=c=black:s=1920x1080:d=30", str(black)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    subprocess.run([ffmpeg_path, "-y", "-i", str(black), "-i", str(audio_mp3), "-shortest", "-c:v", "libx264", "-c:a", "aac", "-b:a", "192k", str(out_mp4)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except Exception as e:
                st.error(f"รวมวิดีโอล้มเหลว: {e}")
                st.stop()

            # 4) Show & download
            st.success("สำเร็จ! ดาวน์โหลดไฟล์ด้านล่าง")
            with open(out_mp4, "rb") as f:
                st.download_button("⬇️ ดาวน์โหลด MP4", f, file_name=filename, mime="video/mp4")

            st.video(str(out_mp4))

st.markdown("""---
**Tips**  
- Pexels เร็วและง่าย เหมาะกับงานสั้น ๆ  
- HuggingFace Inference API คุณภาพขึ้นกับโมเดล อาจมีคิว/หน่วง ถ้าไม่มี token ใช้โหมดอัปโหลดหรือ Pexels ก่อน
- ถ้าดาวน์โหลดไม่เริ่ม ให้คลิกปุ่ม ⬇️ ดาวน์โหลด หรือคลิกสามจุดที่ player → Download
""")    

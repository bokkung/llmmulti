import streamlit as st
import base64
from openai import OpenAI
from PIL import Image
from io import BytesIO
import torch
from diffusers import StableDiffusionPipeline

st.set_page_config(
    page_title="LLM Multimode",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=st.secrets["OPENROUTER_API_KEY"]
    )
except KeyError:
    st.error("❌ โปรดตั้งค่า 'OPENROUTER_API_KEY' ใน Streamlit Secrets เพื่อใช้งานฟังก์ชัน Chat")
    client = None

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

st.markdown("""
<style>
* { margin:0; padding:0; box-sizing:border-box; }
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f172a 0%, #1a1f3a 50%, #0f172a 100%) !important;
    color: #f1f5f9 !important;
}
[data-testid="stSidebar"] { background: rgba(30,41,59,0.8) !important; }
input[type="text"] {
    background-color: rgba(30,41,59,0.9) !important;
    border: 1.5px solid rgba(99,102,241,0.4) !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
    padding: 10px 16px !important;
    transition: all 0.3s ease !important;
}
input[type="text"]:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.2) !important;
    background-color: rgba(99,102,241,0.05) !important;
}
button {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    color:white !important; border:none !important; border-radius:10px !important;
    font-weight:600 !important; padding:12px 20px !important;
    transition: all 0.3s ease !important; box-shadow:0 4px 15px rgba(99,102,241,0.3) !important;
    cursor:pointer !important; height:48px;
}
button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99,102,241,0.5) !important;
}
hr { border-color: rgba(99,102,241,0.2) !important; }
[data-testid="stFileUploader"] {
    padding-top: 0px !important;
}
[data-testid="stFileUploader"] > div > div:first-child {
    height: 48px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
    color: white !important;
    border-radius: 10px !important;
    padding: 10px 16px !important;
    font-weight: 600 !important;
    box-shadow:0 4px 15px rgba(99,102,241,0.3) !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
}
[data-testid="stFileUploader"] > div > div:first-child:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99,102,241,0.5) !important;
}
[data-testid="stFileUploader"] label {
    display: none;
}
[data-testid="stFileUploaderDropzone"] section {
    display: none;
}
[data-testid="stFileUploaderDropzone"] {
    min-height: 0px !important;
    border: none !important;
    padding: 0px !important;
}
</style>
""", unsafe_allow_html=True)

page = st.sidebar.radio("เลือกหน้า", ["Home", "Chat", "Generate Image"])

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful multimodal assistant. You can analyze images provided by the user and respond to text queries."}]
if "uploaded_image_bytes" not in st.session_state:
    st.session_state.uploaded_image_bytes = None
if "last_uploaded_image_display" not in st.session_state:
    st.session_state.last_uploaded_image_display = None

if page == "Home":
    st.markdown("""
    <div style='display:flex;justify-content:center;align-items:center;min-height:80vh; flex-direction:column;'>
        <div style='text-align:center;background:linear-gradient(135deg, rgba(99,102,241,0.1), rgba(236,72,153,0.1));
                    border:1px solid rgba(99,102,241,0.3);border-radius:20px;padding:60px 40px;max-width:600px;
                    backdrop-filter:blur(10px);box-shadow:0 8px 32px rgba(0,0,0,0.3);'>
            <h1 style='font-size:3.5rem;font-weight:700;margin-bottom:20px;
                       color:#ec4899; /* สีสำรองทึบ (ชมพู) */
                       background:linear-gradient(135deg,#6366f1,#ec4899);
                       -webkit-background-clip:text;
                       -webkit-text-fill-color:transparent;'>
                Welcome
            </h1>
            <p style='font-size:1.2rem;color:#cbd5e1;margin:20px 0;line-height:1.6;'>
                ยินดีต้อนรับสู่ LLM Multimode
            </p>
            <p style='font-size:1.2rem;color:#cbd5e1;margin:20px 0;line-height:1.6;'>
                mini project นี้ถูกจัดทำตอนที่ผมกำลังฝึกงานที่NECTEC
            </p>
            <div style="margin-top:20px; display:flex; justify-content:center; gap:20px;">
                <a href="https://github.com/bokkung" target="_blank">
                    <img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/github.svg"
                         alt="GitHub" style="width:40px;height:40px;filter:invert(1);">
                </a>
                <a href="https://hub.docker.com/u/bokkkk51" target="_blank">
                    <img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/docker.svg"
                         alt="Docker Hub" style="width:40px;height:40px;filter:invert(1);">
                </a>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif page == "Chat":
    st.markdown("""
    <div style='background:linear-gradient(135deg, rgba(99,102,241,0.1), rgba(236,72,153,0.1));
                 border-bottom:1px solid rgba(99,102,241,0.3);padding:30px 20px;border-radius:16px;
                 margin-bottom:30px;backdrop-filter:blur(10px);'>
    """, unsafe_allow_html=True)
    # **ใช้ st.markdown ที่แยกสีแทนการใช้ Gradient ที่มีปัญหา**
    st.markdown(f"<h1 style='font-size:2.5rem;margin:0;'><span style='color:#6366f1;'>💬 Multimodal</span> <span style='color:#ec4899;'>Chatbot</span></h1>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


    col1, col2, col3 = st.columns([0.75, 0.125, 0.125])
    with col1:
        user_input = st.text_input("_", key="chat_input", placeholder="พิมพ์ข้อความและ/หรืออัปโหลดรูปภาพ", label_visibility="collapsed")
    with col2:
        uploaded_file = st.file_uploader("📷", type=["jpg","jpeg","png"], key="chat_image_uploader", label_visibility="collapsed")
    with col3:
        send_pressed = st.button("ส่ง", key="chat_send_button", use_container_width=True)

    st.divider()

    for msg in st.session_state.messages:
        if msg["role"] == "system": continue
        is_user = msg["role"] == "user"
        role = " You" if is_user else " Assistant"

        if isinstance(msg["content"], list):
            text_content = ""
            for item in msg["content"]:
                if item["type"] == "text":
                    text_content = item["text"]
                elif item["type"] == "image_url":
                    image_data = item["image_url"]["url"].split(",")[1]
                    st.markdown(f"""
                        <div style='display:flex;justify-content:flex-end;margin:15px 0;'>
                            <div style='background:rgba(99,102,241,0.1);padding:10px;border-radius:10px;max-width:300px;box-shadow:0 4px 10px rgba(99,102,241,0.1);'>
                                <img src='data:image/jpeg;base64,{image_data}' style='width:100%;border-radius:8px;'/>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
            content_display = text_content.strip()
        else:
            content_display = msg["content"].strip()

        if is_user:
            is_default_text = content_display == "ช่วยวิเคราะห์รูปภาพนี้หน่อย" and isinstance(msg["content"], list) and len(msg["content"]) > 1
            if not is_default_text or (is_default_text and len(msg["content"]) == 1):
                st.markdown(f"""
                <div style='display:flex;justify-content:flex-end;margin:15px 0;'>
                    <div style='background:linear-gradient(135deg, rgba(99,102,241,0.3), rgba(99,102,241,0.2));
                                border-left:3px solid #6366f1;color:#f1f5f9;padding:14px 18px;border-radius:14px;
                                max-width:70%;word-wrap:break-word;box-shadow:0 4px 15px rgba(99,102,241,0.2);'>
                        <b style='color:#a78bfa;'>{role}:</b> {content_display}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='display:flex;justify-content:flex-start;margin:15px 0;'>
                <div style='background:linear-gradient(135deg, rgba(236,72,153,0.15), rgba(99,102,241,0.1));
                                border-left:3px solid #ec4899;color:#f1f5f9;padding:14px 18px;border-radius:14px;
                                max-width:70%;word-wrap:break-word;box-shadow:0 4px 15px rgba(236,72,153,0.2);'>
                    <b style='color:#f472b6;'>{role}:</b> {content_display}
                </div>
            </div>
            """, unsafe_allow_html=True)

    if send_pressed and (user_input or uploaded_file):
        if client is None:
            st.error("❌ ไม่สามารถประมวลผลได้เนื่องจากไม่มี OPENROUTER_API_KEY")
            st.stop()

        user_message_content = []
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            base64_image = image_to_base64(image)
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
            user_message_content.append(image_content)
            st.session_state.last_uploaded_image_display = image_content

        text_content = user_input if user_input else "ช่วยวิเคราะห์รูปภาพนี้หน่อย"
        user_message_content.append({
            "type": "text",
            "text": text_content
        })

        st.session_state.messages.append({"role": "user", "content": user_message_content})
        api_messages = st.session_state.messages[:]

        reply_text = ""
        placeholder = st.empty()

        try:
            stream_response = client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=api_messages,
                stream=True
            )

            for chunk in stream_response:
                delta_content = chunk.choices[0].delta.content
                if delta_content:
                    reply_text += delta_content
                    placeholder.markdown(f"""
                    <div style='display:flex;justify-content:flex-start;margin:15px 0;'>
                        <div style='background:linear-gradient(135deg, rgba(236,72,153,0.15), rgba(99,102,241,0.1));
                                        border-left:3px solid #ec4899;color:#f1f5f9;padding:14px 18px;border-radius:14px;
                                        max-width:70%;word-wrap:break-word;box-shadow:0 4px 15px rgba(236,72,153,0.2);'>
                            <b style='color:#f472b6;'> Assistant:</b> {reply_text}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            st.session_state.messages.append({"role":"assistant","content":reply_text})
            st.session_state.uploaded_image_bytes = None
            st.rerun()

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการเรียก API: {e}")
            if st.session_state.messages and st.session_state.messages[-1]["role"]=="user":
                st.session_state.messages.pop()

elif page == "Generate Image":
    st.markdown("""
    <div style='background:linear-gradient(135deg, rgba(99,102,241,0.1), rgba(236,72,153,0.1));
                 border-bottom:1px solid rgba(99,102,241,0.3);padding:30px 20px;border-radius:16px;
                 margin-bottom:30px;backdrop-filter:blur(10px);'>
    """, unsafe_allow_html=True)
    # **ใช้ st.markdown ที่แยกสีแทนการใช้ Gradient ที่มีปัญหา**
    st.markdown(f"<h1 style='font-size:2.5rem;margin:0;'><span style='color:#6366f1;'>🎨 Generate</span> <span style='color:#ec4899;'>Image</span></h1>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    prompt = st.text_input("_", placeholder="พิมพ์ prompt เพื่อสร้างรูปภาพ (English Only)", label_visibility="collapsed")
    generate_btn = st.button(" Generate Image", use_container_width=True)

    if generate_btn and prompt:
        with st.spinner(" กำลังสร้างภาพ... "):
            try:
                if "pipe" not in st.session_state:
                    st.session_state.pipe = StableDiffusionPipeline.from_pretrained(
                        "runwayml/stable-diffusion-v1-5",
                        safety_checker=None
                    )
                    st.session_state.pipe = st.session_state.pipe.to("cpu")

                image = st.session_state.pipe(prompt).images[0]
                st.image(image, caption=prompt, use_container_width=True)
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาด: {e}")
import streamlit as st
import base64
from openai import OpenAI
from PIL import Image
from io import BytesIO
import json
from datetime import datetime
from diffusers import StableDiffusionPipeline
import os

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
    st.error("โŒ ไม่พบ 'OPENROUTER_API_KEY' ใน Streamlit Secrets เพื่อใช้งาน Chat")
    client = None

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

CONVERSATIONS_FILE = "conversations.json"
GEN_HISTORY_FILE = "gen_history.json"

def load_conversations_from_file():
    """โหลดประวัติจาก JSON file"""
    if os.path.exists(CONVERSATIONS_FILE):
        try:
            with open(CONVERSATIONS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                print(f"✅ Loaded {len(data)} conversations")
                return data
        except Exception as e:
            print(f"❌ Error loading: {e}")
            return []
    print(f"⚠️  {CONVERSATIONS_FILE} not found")
    return []

def save_conversations_to_file(conversations):
    """บันทึกประวัติลง JSON file"""
    try:
        with open(CONVERSATIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(conversations, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved to {CONVERSATIONS_FILE}")
    except Exception as e:
        st.error(f"Error saving conversations: {e}")
        print(f"❌ Error: {e}")

def load_gen_history_from_file():
    """โหลดประวัติสร้างรูปจาก JSON file"""
    if os.path.exists(GEN_HISTORY_FILE):
        try:
            with open(GEN_HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

def save_gen_history_to_file(history):
    """บันทึกประวัติสร้างรูปลง JSON file"""
    try:
        with open(GEN_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Error saving gen history: {e}")

def create_new_chat():
    """สร้างแชทใหม่"""
    chat_id = f"chat_{datetime.now().timestamp()}"
    new_chat = {
        "id": chat_id,
        "name": "New Chat",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "messages": [{"role": "system", "content": "You are a helpful multimodal assistant. You can analyze images provided by the user and respond to text queries."}]
    }
    st.session_state.conversations.insert(0, new_chat)
    st.session_state.current_chat_id = chat_id
    save_conversations_to_file(st.session_state.conversations)
    st.rerun()

def delete_chat(chat_id):
    """ลบแชท"""
    st.session_state.conversations = [c for c in st.session_state.conversations if c["id"] != chat_id]
    if st.session_state.current_chat_id == chat_id:
        st.session_state.current_chat_id = st.session_state.conversations[0]["id"] if st.session_state.conversations else None
    save_conversations_to_file(st.session_state.conversations)
    st.rerun()

def rename_chat(chat_id, new_name):
    """เปลี่ยนชื่อแชท"""
    for chat in st.session_state.conversations:
        if chat["id"] == chat_id:
            chat["name"] = new_name
            break
    save_conversations_to_file(st.session_state.conversations)
    st.rerun()

def get_current_chat():
    """ได้แชทปัจจุบัน"""
    for chat in st.session_state.conversations:
        if chat["id"] == st.session_state.current_chat_id:
            return chat
    return None

# Initialize session state
if "conversations" not in st.session_state:
    st.session_state.conversations = load_conversations_from_file()
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = st.session_state.conversations[0]["id"] if st.session_state.conversations else None
if "gen_history" not in st.session_state:
    st.session_state.gen_history = load_gen_history_from_file()

st.markdown("""
<style>
* { margin:0; padding:0; box-sizing:border-box; }
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f172a 0%, #1a1f3a 50%, #0f172a 100%) !important;
    color: #f1f5f9 !important;
}
[data-testid="stSidebar"] { background: rgba(30,41,59,0.8) !important; }
input[type="text"], textarea {
    background-color: rgba(30,41,59,0.9) !important;
    border: 1.5px solid rgba(99,102,241,0.4) !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
    padding: 10px 16px !important;
    transition: all 0.3s ease !important;
}
input[type="text"]:focus, textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.2) !important;
    background-color: rgba(99,102,241,0.05) !important;
}
button {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    color:white !important; border:none !important; border-radius:10px !important;
    font-weight:600 !important; padding:12px 20px !important;
    transition: all 0.3s ease !important; box-shadow:0 4px 15px rgba(99,102,241,0.3) !important;
    cursor:pointer !important;
}
button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99,102,241,0.5) !important;
}
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
    display: none !important;
}
.chat-item {
    background: rgba(99,102,241,0.1);
    border-left: 3px solid #6366f1;
    padding: 10px 12px;
    border-radius: 8px;
    margin: 8px 0;
    cursor: pointer;
    transition: all 0.2s ease;
}
.chat-item:hover {
    background: rgba(99,102,241,0.2);
    transform: translateX(2px);
}
.chat-item.active {
    background: linear-gradient(135deg, rgba(99,102,241,0.3), rgba(236,72,153,0.3));
    border-left-color: #ec4899;
}
</style>
""", unsafe_allow_html=True)

page = st.sidebar.radio("เลือกหน้า", ["Home", "Chat", "Generate Image"])

if page == "Home":
    st.markdown("""
    <div style='display:flex;justify-content:center;align-items:center;min-height:80vh; flex-direction:column;'>
        <div style='text-align:center;background:linear-gradient(135deg, rgba(99,102,241,0.1), rgba(236,72,153,0.1));
                    border:1px solid rgba(99,102,241,0.3);border-radius:20px;padding:60px 40px;max-width:600px;
                    backdrop-filter:blur(10px);box-shadow:0 8px 32px rgba(0,0,0,0.3);'>
            <h1 style='font-size:3.5rem;font-weight:700;margin-bottom:20px;
                       color:#ec4899;
                       background:linear-gradient(135deg,#6366f1,#ec4899);
                       -webkit-background-clip:text;
                       -webkit-text-fill-color:transparent;'>
                Welcome
            </h1>
            <p style='font-size:1.2rem;color:#cbd5e1;margin:20px 0;line-height:1.6;'>
                ยินดีต้อนรับสู่ LLM Multimode
            </p>
            <p style='font-size:1.2rem;color:#cbd5e1;margin:20px 0;line-height:1.6;'>
                mini project นี้ถูกจัดทำตอนที่ผมกำลังฝึกงานที่ NECTEC
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
        <h1 style='font-size:2.5rem;margin:0;'><span style='color:#6366f1;'>💬 Chat</span> <span style='color:#ec4899;'>bot</span></h1>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("###  ประวัติการสนทนา")
        
        if st.button(" สร้างแชทใหม่", use_container_width=True, key="new_chat"):
            create_new_chat()
        
        st.divider()
        
        for chat in st.session_state.conversations:
            col1, col2, col3 = st.columns([0.7, 0.15, 0.15])
            
            with col1:
                if st.button(
                    f" {chat['name'][:20]}...\n_{len(chat['messages'])-1} messages_",
                    key=f"select_{chat['id']}",
                    use_container_width=True,
                    type="secondary" if st.session_state.current_chat_id == chat['id'] else "primary"
                ):
                    st.session_state.current_chat_id = chat['id']
                    st.rerun()
            
            with col2:
                if st.button("✏️", key=f"edit_{chat['id']}", use_container_width=True):
                    st.session_state[f"editing_{chat['id']}"] = True
                    st.rerun()
            
            with col3:
                if st.button("🗑️", key=f"delete_{chat['id']}", use_container_width=True):
                    delete_chat(chat['id'])

            if st.session_state.get(f"editing_{chat['id']}", False):
                new_name = st.text_input(
                    "ชื่อใหม่",
                    value=chat['name'],
                    key=f"rename_{chat['id']}"
                )
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("💾 บันทึก", key=f"save_{chat['id']}", use_container_width=True):
                        rename_chat(chat['id'], new_name)
                        st.session_state[f"editing_{chat['id']}"] = False
                with col2:
                    if st.button("❌ ยกเลิก", key=f"cancel_{chat['id']}", use_container_width=True):
                        st.session_state[f"editing_{chat['id']}"] = False

    current_chat = get_current_chat()
    
    if current_chat:
        # MESSAGES AREA (scrollable)
        for message in current_chat["messages"]:
            if message["role"] == "system":
                continue
            
            is_user = message["role"] == "user"
            
            if isinstance(message["content"], list):
                text_content = ""
                for item in message["content"]:
                    if item.get("type") == "text":
                        text_content = item.get("text", "")
                    elif item.get("type") == "image_url":
                        image_data = item["image_url"]["url"].split(",")[1]
                        st.markdown(f"""
                            <div style='display:flex;justify-content:flex-end;margin:15px 0;'>
                                <div style='background:rgba(99,102,241,0.1);padding:10px;border-radius:10px;max-width:300px;box-shadow:0 4px 10px rgba(99,102,241,0.1);'>
                                    <img src='data:image/jpeg;base64,{image_data}' style='width:100%;border-radius:8px;'/>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
            else:
                text_content = message["content"]
            
            if text_content:
                if is_user:
                    st.markdown(f"""
                    <div style='display:flex;justify-content:flex-end;margin:15px 0;'>
                        <div style='background:linear-gradient(135deg, rgba(99,102,241,0.3), rgba(99,102,241,0.2));
                                    border-left:3px solid #6366f1;color:#f1f5f9;padding:14px 18px;border-radius:14px;
                                    max-width:70%;word-wrap:break-word;box-shadow:0 4px 15px rgba(99,102,241,0.2);'>
                            <b style='color:#a78bfa;'> You:</b> {text_content}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style='display:flex;justify-content:flex-start;margin:15px 0;'>
                        <div style='background:linear-gradient(135deg, rgba(236,72,153,0.15), rgba(99,102,241,0.1));
                                    border-left:3px solid #ec4899;color:#f1f5f9;padding:14px 18px;border-radius:14px;
                                    max-width:70%;word-wrap:break-word;box-shadow:0 4px 15px rgba(236,72,153,0.2);'>
                            <b style='color:#f472b6;'>🤖 Assistant:</b> {text_content}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        
        # INPUT AREA (ล้อมรอบ เหมือน ChatGPT)
        st.markdown("""
        <style>
        .input-container {
            background: linear-gradient(135deg, rgba(99,102,241,0.1), rgba(99,102,241,0.05));
            border: 1.5px solid rgba(99,102,241,0.3);
            border-radius: 15px;
            padding: 15px;
            margin: 10px 0;
            backdrop-filter: blur(10px);
        }
        </style>
        <div class="input-container"></div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([0.75, 0.12, 0.12])
        with col1:
            user_input = st.text_input("_", key="chat_input", placeholder="พิมพ์ข้อความและ/หรืออัปโหลดรูปภาพ", label_visibility="collapsed")
        with col2:
            uploaded_file = st.file_uploader("📷", type=["jpg","jpeg","png"], key="chat_image_uploader", label_visibility="collapsed")
        with col3:
            send_pressed = st.button("ส่ง", key="chat_send_button", use_container_width=True)

        if send_pressed and (user_input or uploaded_file):
            if client is None:
                st.error("โŒ ไม่สามารถประมวลผลได้เนื่องจากไม่มี OPENROUTER_API_KEY")
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

            text_content = user_input if user_input else "ช่วยวิเคราะห์รูปภาพนี้หน่อย"
            user_message_content.append({
                "type": "text",
                "text": text_content
            })

            current_chat["messages"].append({"role": "user", "content": user_message_content})
            
            if current_chat["name"] == "New Chat":
                current_chat["name"] = text_content[:30]

            api_messages = current_chat["messages"][:]
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
                                <b style='color:#f472b6;'>🤖 Assistant:</b> {reply_text}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                current_chat["messages"].append({"role":"assistant","content":reply_text})
                save_conversations_to_file(st.session_state.conversations)
                st.rerun()

            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการเรียก API: {e}")
                if current_chat["messages"] and current_chat["messages"][-1]["role"]=="user":
                    current_chat["messages"].pop()
    else:
        st.markdown("""
        <div style='display:flex;justify-content:center;align-items:center;min-height:60vh; flex-direction:column;'>
            <div style='text-align:center;'>
                <h2 style='color:#cbd5e1;margin:20px 0;'>ยังไม่มีแชท</h2>
                <p style='color:#94a3b8;'>สร้างแชทใหม่เพื่อเริ่มต้นการสนทนา</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

elif page == "Generate Image":
    st.markdown("""
    <div style='background:linear-gradient(135deg, rgba(99,102,241,0.1), rgba(236,72,153,0.1));
                 border-bottom:1px solid rgba(99,102,241,0.3);padding:30px 20px;border-radius:16px;
                 margin-bottom:30px;backdrop-filter:blur(10px);'>
        <h1 style='font-size:2.5rem;margin:0;'><span style='color:#6366f1;'>🎨 Generate</span> <span style='color:#ec4899;'>Image</span></h1>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("###  ประวัติ Prompts ที่สร้าง")
        
        if st.button("🗑️ ล้างประวัติ", key="clear_gen", use_container_width=True):
            st.session_state.gen_history = []
            save_gen_history_to_file(st.session_state.gen_history)
            st.rerun()
        
        st.divider()
        
        if st.session_state.gen_history:
            for i, item in enumerate(st.session_state.gen_history[-10:], 1):
                prompt_preview = item.get("prompt", "")[:40]
                timestamp = item.get("timestamp", "")
                
                st.caption(f"**{i}. {prompt_preview}...**")
                st.caption(f"_{timestamp}_")
                
                if st.button("🔄 ใช้ prompt นี้", key=f"use_prompt_{i}", use_container_width=True):
                    st.session_state.gen_input = item.get("prompt", "")
                    st.session_state.auto_generate = True
                    st.rerun()
        else:
            st.info("ยังไม่มีประวัติการสร้าง")

    if "gen_input" not in st.session_state:
        st.session_state.gen_input = ""
    if "auto_generate" not in st.session_state:
        st.session_state.auto_generate = False

    prompt = st.text_input(
        "_",
        placeholder="พิมพ์ prompt เพื่อสร้างรูปภาพ (English Only)",
        label_visibility="collapsed",
        value=st.session_state.gen_input,
        key="gen_prompt_input"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        generate_btn = st.button(" Generate Image", use_container_width=True)

    if (generate_btn or st.session_state.auto_generate) and prompt:
        with st.spinner(" กำลังสร้างรูปภาพ..."):
            try:
                if "pipe" not in st.session_state:
                    st.session_state.pipe = StableDiffusionPipeline.from_pretrained(
                        "runwayml/stable-diffusion-v1-5",
                        safety_checker=None
                    )
                    st.session_state.pipe = st.session_state.pipe.to("cpu")

                image = st.session_state.pipe(prompt).images[0]
                st.image(image, caption=prompt, use_container_width=True)
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.gen_history.append({
                    "prompt": prompt,
                    "timestamp": timestamp
                })
                save_gen_history_to_file(st.session_state.gen_history)
                
                st.session_state.gen_input = ""
                st.session_state.auto_generate = False
                
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาด: {e}")
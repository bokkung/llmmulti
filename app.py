import streamlit as st
import base64
from openai import OpenAI
from PIL import Image
from io import BytesIO
import torch
from diffusers import StableDiffusionPipeline


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=st.secrets["OPENROUTER_API_KEY"]
)


custom_css = """
<style>

[data-testid="stFileUploaderDropzone"] > div:nth-child(2) {
    display: none !important;
}


[data-testid="stFileUploader"] > div:nth-child(2) > div:nth-child(2) {
    display: none !important;
}


[data-testid="stFileUploader"] {
    max-width: 120px; 
}


[data-testid="stFileUploaderDropzone"] {
    min-height: 40px !important; 
    height: 40px !important; 
    padding: 0px !important; 
    border: 1px solid #333333; 
    border-radius: 0.5rem;
}


[data-testid="stFileUploaderDropzone"] > div:nth-child(1) {
    display: flex;
    justify-content: center !important; 
    align-items: center; 
    height: 100%;
    width: 100%; 
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)




page = st.sidebar.selectbox("เลือกหน้า", ["Home", "Chat", "Generate Image"])






if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]
if "last_uploaded_image" not in st.session_state:
    st.session_state.last_uploaded_image = None
    
    
    
    
if page == "Home":
    st.title("Welcome to my llm multimode")
    st.write("mini project นี้ถูกจัดทำตอนที่ผมกำลังฝึกงานที่NECTEC")
    st.stop()  



elif page  == "Chat":
    st.title("chatbot")
    col1, col2, col3 = st.columns([0.7, 0.15, 0.15])
    with col1:
        user_input = st.text_input("_", key="chat_input", placeholder="พิมพ์ข้อความหรือคำอธิบายรูปภาพ", label_visibility="collapsed")
    with col2:
        uploaded_file = st.file_uploader("Upload", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    with col3:
        send_pressed = st.button("ส่ง", use_container_width=True)

    if send_pressed and (user_input or uploaded_file):
        api_messages = st.session_state.messages[:]
        user_content_for_api = []

        if uploaded_file:
            image_data = uploaded_file.read()
            encoded_image = base64.b64encode(image_data).decode("utf-8")
            st.session_state.last_uploaded_image = image_data

            image_text = user_input if user_input else "โปรดอธิบายรูปภาพนี้"
            user_content_for_api.append({"type": "text", "text": image_text})
            user_content_for_api.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}})

            display_text = "[รูปภาพ]"
            if user_input:
                display_text += " " + user_input
            st.session_state.messages.append({"role": "user", "content": display_text})

        elif user_input:
            user_content_for_api.append({"type": "text", "text": user_input})
            st.session_state.messages.append({"role": "user", "content": user_input})

        if user_content_for_api:
            api_messages.append({"role": "user", "content": user_content_for_api})

        try:
            response = client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=api_messages
            )
            reply = response.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาด: {e}")
            if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                st.session_state.messages.pop()

    
    for msg in st.session_state.messages:
        if msg["role"] == "system":
            continue
        role = "คุณ" if msg["role"] == "user" else "OpenRouter"
        if msg["role"] == "user":
            bg_color = "#000000"; text_color = "#FFFFFF"; align = "right"; margin = "margin-left:auto;"
        else:
            bg_color = "#000000"; text_color = "#FFFFFF"; align = "left"; margin = "margin-right:auto"

        if "[รูปภาพ]" in msg["content"]:
            st.markdown(f"""
                <div style='background-color:{bg_color}; color:{text_color}; padding:10px 15px; border-radius:10px; margin:5px 0; text-align:{align}; width:fit-content; max-width:70%; {margin}'>
                    <b>{role}:</b> {msg['content'].replace("[รูปภาพ]", "").strip()}
                </div>""", unsafe_allow_html=True)
            if st.session_state.last_uploaded_image and msg["role"] == "user":
                st.image(st.session_state.last_uploaded_image, width=250)
        else:
            st.markdown(f"""
                <div style='background-color:{bg_color}; color:{text_color}; padding:10px 15px; border-radius:10px; margin:5px 0; text-align:{align}; width:fit-content; max-width:70%; {margin}'>
                    <b>{role}:</b> {msg['content']}
                </div>""", unsafe_allow_html=True)


elif page == "Generate Image":
    st.title("Generate Image")
    
    
    prompt = st.text_input("_", placeholder="พิมพ์ prompt เพื่อสร้างรูปภาพ(Eng Only)",label_visibility="collapsed")
    generate_btn = st.button("Generate Image")

    if generate_btn and prompt:
        with st.spinner("กำลังสร้างภาพ... ใช้เวลาประมาณ(10 นาที เพราะผมรันด้วย cpu )"):
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
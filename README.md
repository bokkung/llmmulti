# llmmulti
## Deploy link url https://llmmulti.streamlit.app/

# docker วิธีใช้ 
docker build -t llm-multimode .
docker run -p 8501:8501 -e OPENROUTER_API_KEY=your_api_key llm-multimode
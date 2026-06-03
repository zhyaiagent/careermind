"""CareerMind"""
import json, uuid
import streamlit as st
import requests, os

API_BASE = os.environ.get("JOBSENSE_API_URL", "http://localhost:8001")
st.set_page_config(page_title="CareerMind", page_icon="", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', 'Microsoft YaHei', sans-serif; }
    .stApp {
        background: linear-gradient(135deg, #f8f9ff, #fdf7f2, #f2f9f8, #f7f4fc, #f8f9ff);
        background-size: 400% 400%;
        animation: bgFlow 25s ease infinite;
    }
    @keyframes bgFlow {
        0% { background-position: 0% 50%; }
        25% { background-position: 100% 0%; }
        50% { background-position: 100% 100%; }
        75% { background-position: 0% 100%; }
        100% { background-position: 0% 50%; }
    }
    .hero { text-align: center; margin: 1.5rem 0 1rem 0; }
    .hero-title { font-size: 6rem !important; font-weight: 800; letter-spacing: -2px; margin: 0; line-height: 1.1; }
    .hero-title .c { color: #1e1e2f; } .hero-title .m { color: #5b5fe3; }
    .hero-sub { font-size: 0.75rem; color: #8e8ea0; margin-top: 0.4rem; }
    .tag-row {
        display: flex; justify-content: center; flex-wrap: wrap;
        gap: 0.5rem; margin: 0.8rem 0 0.5rem 0;
    }
    .tag {
        font-size: 0.65rem; font-weight: 500; color: #555;
        background: #fff; border: 1px solid #e5e5ed;
        border-radius: 20px; padding: 0.25rem 0.8rem;
        white-space: nowrap;
    }
    .stButton > button {
        border-radius: 6px; border: 1px solid #e0e0e8; background: #fff;
        color: #333; font-size: 0.78rem; font-weight: 500;
    }
    .stButton > button:hover { background: #f5f5ff; border-color: #5b5fe3; color: #5b5fe3; }
    .stFileUploader > div { border: 1.5px dashed #d5d5e0 !important; border-radius: 8px !important; }
    .stFileUploader > div:hover { border-color: #5b5fe3 !important; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state: st.session_state.messages = []
if "thread_id" not in st.session_state: st.session_state.thread_id = str(uuid.uuid4())[:8]

st.markdown("""
<div class="hero">
    <p class="hero-title"><span class="c">Career</span><span class="m">Mind</span></p>
    <p class="hero-sub">Your AI Career Agent &mdash; Intelligent Job Search, Analysis &amp; Automation</p>
    <div class="tag-row">
        <span class="tag">Hybrid Agent</span>
        <span class="tag">ReAct + Plan-Execute</span>
        <span class="tag">RAG (Vector+BM25+RRF+Reranker)</span>
        <span class="tag">MCP Protocol</span>
        <span class="tag">8 Tools</span>
        <span class="tag">Browser Automation</span>
        <span class="tag">Edge Playwright</span>
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("###   系统状态")
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=3)
        if resp.status_code == 200:
            d = resp.json()
            st.success("  API 在线")
            agent_text = d.get("agent", "?").replace("Hybrid(ReAct+PlanExecute)", "ReAct + Plan-Execute")
            st.markdown(f"**架构**：{agent_text}")
            st.markdown(f"**工具**：{d.get('tools',0)} 个内置 + {d.get('mcp_tools',0)} 个 MCP")
        else:
            st.error("  API 离线")
    except Exception:
        st.warning("  API 未连接")

    st.divider()
    st.markdown("###   知识库")
    f = st.file_uploader("上传文档", type=["pdf","docx","txt"], help="支持 PDF / Word / 文本")
    if f and st.button("  上传处理", use_container_width=True):
        try:
            import base64
            c = base64.b64encode(f.read()).decode()
            r = requests.post(f"{API_BASE}/upload", json={"file_content":c,"file_name":f.name,"file_type":f.name.split(".")[-1]})
            st.success(r.json().get("message","OK")) if r.ok else st.error(r.text)
        except Exception as e: st.error(str(e))

    st.divider()
    if st.button("  清除对话", use_container_width=True):
        st.session_state.messages = []; st.session_state.thread_id = str(uuid.uuid4())[:8]; st.rerun()

    st.divider()
    st.caption("  分析JD |   技能匹配 |   薪资查询")
    st.caption("  浏览器搜索 |   联网查询 |   上传文档")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("输入求职问题或浏览器指令..."):
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"): st.markdown(prompt)
    with st.chat_message("assistant"):
        ph = st.empty(); ans = ""
        try:
            resp = requests.post(f"{API_BASE}/chat/stream", json={"message":prompt,"thread_id":st.session_state.thread_id}, stream=True, timeout=120)
            if resp.status_code == 200:
                for line in resp.iter_lines():
                    if not line: continue
                    s = line.decode("utf-8") if isinstance(line, bytes) else line
                    if not s.startswith("data: "): continue
                    try: d = json.loads(s[6:])
                    except: continue
                    t = d.get("type","")
                    if t == "thinking": ph.caption("  思考中...")
                    elif t == "browser": ph.caption("  浏览器操作中...")
                    elif t == "token": ans += d.get("content",""); ph.markdown(ans + "")
                    elif t == "done": ph.markdown(ans)
                    elif t == "error": ph.error(d.get("message",""))
            else: ph.error(f"API 错误: {resp.status_code}")
        except requests.ConnectionError: ph.warning("API 未连接")
        except Exception as e: ph.error(str(e))
        if ans: st.session_state.messages.append({"role":"assistant","content":ans})

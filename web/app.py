"""CareerMind"""
import json, uuid, streamlit as st, requests, os

API_BASE = os.environ.get("JOBSENSE_API_URL", "http://localhost:8001")
st.set_page_config(page_title="CareerMind", page_icon="", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', 'Microsoft YaHei', sans-serif; }
.stApp { background: #fafbfc; }
.hero { text-align: center; margin: 1rem 0 0.5rem 0; }
.hero-title { font-size: 6rem !important; font-weight: 800; letter-spacing: -2px; margin: 0; line-height: 1.1; }
.hero-title .c { color: #1e1e2f; } .hero-title .m { color: #5b5fe3; }
.tag-row { display: flex; justify-content: center; flex-wrap: wrap; gap: 0.5rem; margin: 0.5rem 0; }
.tag { font-size: 0.65rem; font-weight: 500; color: #555; background: #fff; border: 1px solid #e5e5ed; border-radius: 20px; padding: 0.25rem 0.8rem; white-space: nowrap; }
.stButton > button { border-radius: 6px; border: 1px solid #e0e0e8; background: #fff; color: #333; font-size: 0.78rem; }
.stButton > button:hover { background: #f5f5ff; border-color: #5b5fe3; color: #5b5fe3; }
footer { visibility: hidden; }
.stChatMessage { scroll-margin-top: 1rem; }
.stMain { overflow-anchor: none; }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state: st.session_state.messages = []
if "thread_id" not in st.session_state: st.session_state.thread_id = str(uuid.uuid4())[:8]

st.markdown("""
<div class="hero">
    <p class="hero-title"><span class="c">Career</span><span class="m">Mind</span></p>
    <div class="tag-row">
        <span class="tag">Hybrid Agent</span><span class="tag">ReAct + Plan-Execute</span><span class="tag">RAG (Vector+BM25+RRF+Reranker)</span><span class="tag">MCP Protocol</span><span class="tag">8 Tools</span><span class="tag">Browser Automation</span><span class="tag">Edge Playwright</span>
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
            st.markdown(f"**架构**: {d.get('agent','?').replace('Hybrid(ReAct+PlanExecute)','ReAct + Plan-Execute')}")
            st.markdown(f"**工具**: {d.get('tools',0)} 个内置 + {d.get('mcp_tools',0)} 个 MCP")
        else: st.error("  API 离线")
    except Exception: st.warning("  API 未连接")

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

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("输入求职问题或浏览器指令..."):
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        ans = ""
        status = st.status("Thinking...", expanded=False)
        try:
            with requests.post(f"{API_BASE}/chat/stream", json={"message":prompt,"thread_id":st.session_state.thread_id}, stream=True, timeout=120) as resp:
                if resp.status_code == 200:
                    for line in resp.iter_lines():
                        if not line: continue
                        s = line.decode("utf-8") if isinstance(line, bytes) else line
                        if not s.startswith("data: "): continue
                        try: d = json.loads(s[6:])
                        except: continue
                        t = d.get("type","")
                        if t == "thinking": status.update(label="Thinking...", state="running")
                        elif t == "browser": status.update(label="Browser operating...", state="running")
                        elif t == "token": ans += d.get("content","")
                        elif t == "done": break
                        elif t == "error": status.update(label="Error", state="error", expanded=True); st.error(d.get("message",""))
                else: status.update(label=f"API Error {resp.status_code}", state="error")
        except requests.ConnectionError: status.update(label="API offline", state="error")
        except Exception as e: status.update(label=str(e)[:50], state="error")
        status.update(label="Done", state="complete", expanded=False)
        if ans: st.markdown(ans); st.session_state.messages.append({"role":"assistant","content":ans})

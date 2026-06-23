import { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";

const API = "/api";

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState({ online: false, agent: "?", tools: 0, mcp: 0 });
  const [threadId] = useState(() => Math.random().toString(36).slice(2, 10));
  const bottomRef = useRef(null);

  useEffect(() => {
    fetch(`${API}/health`).then(r => r.json()).then(d =>
      setHealth({ online: true, agent: d.agent?.replace("Hybrid(ReAct+PlanExecute)","ReAct + Plan-Execute"), tools: d.tools, mcp: d.mcp_tools })
    ).catch(() => setHealth(h => ({ ...h, online: false })));
  }, []);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  const send = async () => {
    const msg = input.trim(); if (!msg || loading) return;
    setInput(""); setMessages(m => [...m, { role: "user", content: msg }, { role: "assistant", content: "" }]); setLoading(true);
    try {
      const r = await fetch(`${API}/chat/stream`, { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({ message:msg, thread_id:threadId }) });
      const reader = r.body.getReader(); const decoder = new TextDecoder();
      let ans = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split("\n");
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const d = JSON.parse(line.slice(6));
            if (d.type === "token") { ans += d.content; setMessages(m => { const n = [...m]; n[n.length-1] = { role:"assistant", content: ans }; return n; }); }
            if (d.type === "done" || d.type === "error") {}
          } catch {}
        }
      }
    } catch { setMessages(m => { const n = [...m]; n[n.length-1] = { role:"assistant", content: "API offline" }; return n; }); }
    setLoading(false);
  };

  const handleUpload = async (e) => {
    const file = e.target.files?.[0]; if (!file) return;
    const reader = new FileReader();
    reader.onload = async () => {
      const b64 = reader.result.split(",")[1];
      try { await fetch(`${API}/upload`, { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({ file_content:b64, file_name:file.name, file_type:file.name.split(".").pop() }) }); alert("OK"); }
      catch { alert("Upload failed"); }
    };
    reader.readAsDataURL(file);
  };

  return (
    <>
      <aside className="sidebar">
        <h3>系统状态</h3>
        <div className="row"><span><span className={`dot ${health.online ? "green" : "red"}`} /> {health.online ? "API 在线" : "API 离线"}</span></div>
        {health.online && <>
          <div className="row"><span>架构</span><strong>{health.agent}</strong></div>
          <div className="row"><span>工具</span><strong>{health.tools} 内置 + {health.mcp} MCP</strong></div>
        </>}

        <h3>知识库管理</h3>
        <label className="upload"><input type="file" accept=".pdf,.docx,.txt" onChange={handleUpload} />上传文档 (PDF / DOCX / TXT)</label>

        <button className="btn" onClick={() => setMessages([])}>清除对话记忆</button>

        <h3>使用提示</h3>
        <p className="tip">分析JD | 技能匹配 | 薪资查询<br/>浏览器搜索 | 联网查询 | 上传文档</p>
      </aside>

      <main className="chat-area">
        <div className="hero">
          <h1><span className="c">Career</span><span className="m">Mind</span></h1>
          <div className="tags">
            <span>Hybrid Agent</span><span>ReAct + Plan-Execute</span>
            <span>RAG (Vector+BM25+RRF+Reranker)</span><span>MCP Protocol</span>
            <span>8 Tools</span><span>Browser Automation</span><span>Edge Playwright</span>
          </div>
        </div>

        <div className="messages">
          {messages.map((m, i) => (
            <div key={i} className={`msg-row ${m.role}`}>
              <div className={`msg-avatar ${m.role}`}>{m.role === "assistant" ? "" : ""}</div>
              <div className={`msg ${m.role}`}><ReactMarkdown>{m.content}</ReactMarkdown></div>
            </div>
          ))}
          {loading && <div className="loading">思考中...</div>}
          <div ref={bottomRef} />
        </div>

        <div className="input-area">
          <input value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => e.key === "Enter" && send()} placeholder="输入求职问题或浏览器指令..." disabled={loading} />
          <button onClick={send} disabled={loading}>发送</button>
        </div>
      </main>
    </>
  );
}

export default App;

import { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";

const API = "/api";

function Login({ onLogin }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [mode, setMode] = useState("login");
  const [msg, setMsg] = useState("");

  const submit = async () => {
    const ep = mode === "login" ? "login" : "register";
    try {
      const r = await fetch(`${API}/${ep}?username=${username}&password=${password}`, { method: "POST" });
      const d = await r.json();
      if (d.ok) {
        if (mode === "login") { localStorage.setItem("cm_token", d.token); localStorage.setItem("cm_user", d.username); onLogin(d.username, d.token); }
        else { setMsg("Registered! Please login."); setMode("login"); }
      } else { setMsg(d.error || "Error"); }
    } catch { setMsg("API offline"); }
  };

  return (
    <div style={{display:"flex",justifyContent:"center",alignItems:"center",height:"100vh",background:"#f8fafc"}}>
      <div style={{background:"#fff",padding:"2rem",borderRadius:"12px",boxShadow:"0 4px 24px rgba(0,0,0,0.08)",width:"360px",textAlign:"center"}}>
        <h1 style={{fontSize:"2rem",fontWeight:800,marginBottom:"0.5rem"}}>Career<span style={{color:"#5b5fe3"}}>Mind</span></h1>
        <p style={{color:"#94a3b8",marginBottom:"1.5rem",fontSize:"0.85rem"}}>{mode === "login" ? "Sign in" : "Create account"}</p>
        <input value={username} onChange={e => setUsername(e.target.value)} placeholder="Username" style={{width:"100%",padding:"0.6rem",marginBottom:"0.5rem",border:"1px solid #e2e8f0",borderRadius:6,fontSize:"0.9rem"}} />
        <input value={password} onChange={e => setPassword(e.target.value)} type="password" placeholder="Password" onKeyDown={e => e.key==="Enter"&&submit()} style={{width:"100%",padding:"0.6rem",marginBottom:"0.5rem",border:"1px solid #e2e8f0",borderRadius:6,fontSize:"0.9rem"}} />
        <button onClick={submit} style={{width:"100%",padding:"0.6rem",background:"#5b5fe3",color:"#fff",border:"none",borderRadius:6,fontSize:"0.9rem",cursor:"pointer",marginBottom:"0.5rem"}}>{mode === "login" ? "Login" : "Register"}</button>
        {msg && <p style={{color:"#ef4444",fontSize:"0.8rem"}}>{msg}</p>}
        <p style={{fontSize:"0.8rem",color:"#94a3b8",cursor:"pointer"}} onClick={()=>{setMode(mode==="login"?"register":"login");setMsg("")}}>
          {mode === "login" ? "No account? Register" : "Have account? Login"}
        </p>
      </div>
    </div>
  );
}

function App() {
  const [user, setUser] = useState(() => localStorage.getItem("cm_user"));
  const [token, setToken] = useState(() => localStorage.getItem("cm_token"));
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState({ online: false });
  const threadId = user || "default";
  const bottomRef = useRef(null);

  useEffect(() => {
    fetch(`${API}/health`).then(r => r.json()).then(d =>
      setHealth({ online: true, agent: d.agent?.replace("Hybrid(ReAct+PlanExecute)","ReAct+PlanExecute"), tools: d.tools, mcp: d.mcp_tools })
    ).catch(() => {});
    fetch(`${API}/history?thread_id=${threadId}`).then(r => r.json()).then(d => {
      if (d.messages?.length) setMessages(d.messages.map(m => ({role:m.role,content:m.content})));
    }).catch(() => {});
  }, [threadId]);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  if (!user) return <Login onLogin={(u, t) => { setUser(u); setToken(t); }} />;

  const send = async () => {
    const msg = input.trim(); if (!msg || loading) return;
    setInput(""); setMessages(m => [...m, { role: "user", content: msg }, { role: "assistant", content: "" }]); setLoading(true);
    try {
      const r = await fetch(`${API}/chat/stream`, { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({ message:msg, thread_id:threadId }) });
      const reader = r.body.getReader(); const decoder = new TextDecoder(); let ans = "";
      while (true) {
        const { done, value } = await reader.read(); if (done) break;
        const lines = decoder.decode(value, { stream: true }).split("\n");
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try { const d = JSON.parse(line.slice(6)); if (d.type === "token") { ans += d.content; setMessages(m => { const n = [...m]; n[n.length-1] = { role:"assistant", content: ans }; return n; }); } } catch {}
        }
      }
    } catch { setMessages(m => { const n = [...m]; n[n.length-1] = { role:"assistant", content: "API offline" }; return n; }); }
    setLoading(false);
  };

  const logout = () => { localStorage.removeItem("cm_token"); localStorage.removeItem("cm_user"); setUser(null); setToken(null); setMessages([]); };

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
        <h3>System</h3>
        <div className="row"><span><span className={`dot ${health.online ? "green" : "red"}`} /> {health.online ? "API Online" : "Offline"}</span></div>
        {health.online && <>
          <div className="row"><span>Agent</span><strong>{health.agent}</strong></div>
          <div className="row"><span>Tools</span><strong>{health.tools} + {health.mcp} MCP</strong></div>
        </>}
        <div className="row"><span>User</span><strong>{user}</strong></div>
        <h3>KB</h3>
        <label className="upload"><input type="file" accept=".pdf,.docx,.txt" onChange={handleUpload} />Upload PDF/DOCX/TXT</label>
        <button className="btn" onClick={logout}>Logout</button>
        <h3>Tips</h3>
        <p className="tip">JD Analysis | Skill Match | Salary</p>
        <p className="tip">Browser Search | Web Search</p>
      </aside>

      <main className="chat-area">
        <div className="hero">
          <h1><span className="c">Career</span><span className="m">Mind</span></h1>
          <div className="tags">
            <span>Hybrid Agent</span><span>ReAct + Plan-Execute</span>
            <span>RAG</span><span>MCP Protocol</span>
            <span>8 Tools</span><span>Browser Auto</span>
          </div>
        </div>
        <div className="messages">
          {messages.map((m, i) => (
            <div key={i} className={`msg-row ${m.role}`}>
              <div className={`msg-avatar ${m.role}`}>{m.role === "assistant" ? "" : ""}</div>
              <div className={`msg ${m.role}`}><ReactMarkdown>{m.content}</ReactMarkdown></div>
            </div>
          ))}
          {loading && <div className="loading">Thinking...</div>}
          <div ref={bottomRef} />
        </div>
        <div className="input-area">
          <input value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => e.key === "Enter" && send()} placeholder="Ask anything or give browser commands..." disabled={loading} />
          <button onClick={send} disabled={loading}>Send</button>
        </div>
      </main>
    </>
  );
}

export default App;

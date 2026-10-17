import { useEffect, useRef, useState } from "react";

const KEY = "hr-agent-thread";
const newId = () => Array.from({ length: 32 }, () => Math.floor(Math.random() * 16).toString(16)).join("");
const NODE = { router: "Router", faq_rag: "FAQ · RAG", booking: "Booking", chat: "Chat", guardrail: "Guardrail" };
const HINTS = [
  "I need to book a 1-on-1 Performance Review session.",
  "How often are performance reviews conducted?",
  "How many days of PTO do I get?",
];

export default function App() {
  const [threadId, setThreadId] = useState(() => localStorage.getItem(KEY) || newId());
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [booking, setBooking] = useState({ status: "idle" });
  const [lastNode, setLastNode] = useState("");
  const last = useRef("");
  const bottom = useRef(null);

  useEffect(() => {
    localStorage.setItem(KEY, threadId);
  },[threadId]);
  useEffect(() => { bottom.current?.scrollIntoView({ behavior: "smooth" }) }, [messages, loading, error]);

  async function send(text, retry = false) {
    text = text.trim();
    if (!text || loading) return;
    last.current = text;
    setError("");
    setInput("");
    if (!retry) setMessages((m) => [...m, { role: "user", content: text }]);
    setLoading(true);
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, thread_id: threadId }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        const d = data.detail;
        throw new Error(typeof d === "string" ? d : d?.[0]?.msg || `Request failed (${res.status})`);
      }
      setMessages((m) => [...m, { role: "assistant", content: data.reply, citations: data.citations, node: data.node }]);
      setBooking(data.booking);
      setLastNode(data.node);
    } catch (e) {
      setError(e.message || "Could not reach the server.");
    } finally {
      setLoading(false);
    }
  }

  function newChat() {
    setThreadId(newId());
    setMessages([]);
    setBooking({ status: "idle" });
    setLastNode("");
    setError("");
  }

  return (
    <div className="app">
      <header>
        <div>
          <h1>HR Assistant</h1>
          <small>Thread <code>{threadId.slice(0, 8)}</code></small>
        </div>
        <div className="right">
          <span className={"badge " + (loading ? "busy" : "")}>
            {loading ? "Thinking…" : lastNode ? `Last node: ${NODE[lastNode] || lastNode}` : "Idle"}
          </span>
          <span className={"badge b-" + booking.status}>Booking: {booking.status}</span>
          <button onClick={newChat}>New chat</button>
        </div>
      </header>

      <main>
        {messages.length === 0 && (
          <div className="empty">
            <p>Ask about the handbook, or book a session. You can switch between them anytime.</p>
            {HINTS.map((h) => (
              <button key={h} className="hint" onClick={() => send(h)}>{h}</button>
            ))}
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={"msg " + m.role}>
            <p>{m.content}</p>
            {m.citations?.length > 0 && (
              <div className="cites">
                {m.citations.map((c) => (
                  <span key={c.source + c.page}>{c.source.replace(".pdf", "")} · p.{c.page}</span>
                ))}
              </div>
            )}
            {m.node && <small className="via">via {NODE[m.node] || m.node}</small>}
          </div>
        ))}
        {loading && <div className="msg assistant typing">● ● ●</div>}
        {error && (
          <div className="error" role="alert">
            {error} <button onClick={() => send(last.current, true)}>Retry</button>
          </div>
        )}
        <div ref={bottom} />
      </main>

      <form onSubmit={(e) => { e.preventDefault(); send(input); }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type a message…"
          maxLength={2000}
          disabled={loading}
          autoFocus
        />
        <button type="submit" disabled={loading || !input.trim()}>Send</button>
      </form>
    </div>
  );
}

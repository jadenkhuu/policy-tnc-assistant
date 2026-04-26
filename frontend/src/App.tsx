import { useState, useRef, useEffect, Fragment } from "react";
import { queryPolicy, getTopics } from "./api";
import type { QueryResponse, Topic } from "./api";
import "./App.css";

function AnnotatedAnswer({ text }: { text: string }) {
  const parts = text.split(/(\(\d+\))/g);
  return (
    <p className="answer-text">
      {parts.map((part, i) =>
        /^\(\d+\)$/.test(part)
          ? <sup key={i} className="cite-marker">{part}</sup>
          : <Fragment key={i}>{part}</Fragment>
      )}
    </p>
  );
}

function Sidebar({ topics }: { topics: Topic[] }) {
  const [expanded, setExpanded] = useState<string | null>(null);

  function toggle(title: string) {
    setExpanded(prev => prev === title ? null : title);
  }

  return (
    <aside className="sidebar">
      <div className="sidebar-card">
        <div className="sidebar-label">Topics Currently Covered</div>
        <ul className="topic-list">
          {topics.map(topic => (
            <li key={topic.title} className="topic-item">
              <button
                className={`topic-btn ${expanded === topic.title ? "open" : ""}`}
                onClick={() => toggle(topic.title)}
              >
                <span className="topic-title">{topic.title.replace(/^Bing Lee'?s?\s*/i, "")}</span>
                <svg className="topic-chevron" width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true">
                  <path d="M3 4.5l3 3 3-3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </button>
              {expanded === topic.title && (
                <div className="topic-detail">
                  {topic.url ? (
                    <a className="topic-url" href={topic.url} target="_blank" rel="noopener noreferrer">
                      View source ↗
                    </a>
                  ) : (
                    <span className="topic-no-url">No source URL</span>
                  )}
                </div>
              )}
            </li>
          ))}
        </ul>
      </div>
    </aside>
  );
}

export default function App() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [topics, setTopics] = useState<Topic[]>([]);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    getTopics().then(setTopics);
  }, []);

  function autoResize() {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!question.trim() || loading) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await queryPolicy(question.trim());
      setResult(data);
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 80);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as unknown as React.FormEvent);
    }
  }

  return (
    <div className="app">
      <section className="hero">
        <div className="hero-inner">
          <p className="hero-eyebrow">Internal Tool</p>
          <h1 className="hero-title">Bing Lee: Policies and T&amp;Cs Helper</h1>
          <p className="hero-subtitle">
            Ask any question about Bing Lee policies and get instant, cited answers to help your customers.
          </p>
          <p className="hero-disclaimer">
            Policy information is sourced exclusively from Bing Lee's public website — no internal systems, customer data, or POS infrastructure.
            This is a personal portfolio project built on publicly available data only.
          </p>
        </div>
      </section>

      <div className="page-body">
        {topics.length > 0 && <Sidebar topics={topics} />}

        <div className="main-col">
          <div className="inquiry-wrap">
            <form className="inquiry-form" onSubmit={handleSubmit}>
              <textarea
                ref={textareaRef}
                className="inquiry-input"
                value={question}
                onChange={(e) => { setQuestion(e.target.value); autoResize(); }}
                onKeyDown={handleKeyDown}
                placeholder="e.g. 'A customer wants to return a product but its been over 20 days.' or 'A customer missed their delivery window.'"
                disabled={loading}
                rows={3}
              />
              <div className="form-footer">
                <div className="input-hint">Press Enter to submit · Shift+Enter for new line</div>
                <button
                  className="submit-btn"
                  type="submit"
                  disabled={loading || !question.trim()}
                >
                  {loading ? (
                    <span className="btn-content"><span className="spinner" />Searching…</span>
                  ) : (
                    <span className="btn-content">
                      Ask Now
                      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                        <path d="M3 8h10M9 4l4 4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                    </span>
                  )}
                </button>
              </div>
            </form>
          </div>

          {(result || error) && (
            <div className="results" ref={resultsRef}>
              {error && (
                <div className="error-card">
                  <div className="error-icon">!</div>
                  <p>{error}</p>
                </div>
              )}
              {result && (
                <>
                  <div className="answer-card">
                    <div className="card-label">Answer</div>
                    <AnnotatedAnswer text={result.answer} />
                  </div>
                  {result.citations.length > 0 && (
                    <div className="sources-card">
                      <div className="card-label">Sources</div>
                      <ol className="sources-list">
                        {result.citations.map((c, i) => (
                          <li key={i} className="source-item">
                            <span className="source-num">({i + 1})</span>
                            <div className="source-content">
                              <span className="source-title">{c.title}</span>
                              {c.url ? (
                                <a className="source-url" href={c.url} target="_blank" rel="noopener noreferrer">{c.url}</a>
                              ) : (
                                <span className="source-section">{c.section}</span>
                              )}
                            </div>
                          </li>
                        ))}
                      </ol>
                    </div>
                  )}
                </>
              )}
            </div>
          )}
        </div>
      </div>

      <footer className="footer">
        <div className="footer-inner">
          Bing Lee Policy Assistant · Internal Use Only
        </div>
      </footer>
    </div>
  );
}

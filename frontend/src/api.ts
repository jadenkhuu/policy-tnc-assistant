const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:5050";

export interface Citation {
  title: string;
  section: string;
  url: string;
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];
}

export interface Topic {
  title: string;
  url: string;
  section_count: number;
}

export async function getTopics(): Promise<Topic[]> {
  const res = await fetch(`${API_BASE}/topics`);
  if (!res.ok) return [];
  return res.json();
}

export async function queryPolicy(question: string): Promise<QueryResponse> {
  const res = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error ?? "Request failed");
  }

  return res.json();
}

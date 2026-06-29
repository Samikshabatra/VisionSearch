import { useState } from "react";

const EXAMPLES = [
  "a dog running on the beach",
  "two people riding bicycles",
  "a man playing guitar",
  "children playing in a park",
  "a woman in a red dress",
];

export default function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function runQuery(q) {
    if (!q.trim()) return;
    setLoading(true);
    setError("");
    try {
      const res = await fetch("/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, k: 12 }),
      });
      if (!res.ok) throw new Error(`search failed (${res.status})`);
      const data = await res.json();
      setResults(data.results);
    } catch (err) {
      setError(err.message);
      setResults([]);
    } finally {
      setLoading(false);
    }
  }

  function handleSearch(e) {
    e.preventDefault();
    runQuery(query);
  }

  function handleExample(q) {
    setQuery(q);
    runQuery(q);
  }

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100 px-6 py-10">
      <header className="max-w-3xl mx-auto text-center mb-8">
        <h1 className="text-4xl font-bold tracking-tight">VisionSearch</h1>
        <p className="text-neutral-400 mt-2">Type a sentence, get the matching images.</p>
      </header>

      <form onSubmit={handleSearch} className="max-w-2xl mx-auto flex gap-2 mb-4">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="two people on a beach at sunset"
          className="flex-1 rounded-lg bg-neutral-900 border border-neutral-700 px-4 py-3 outline-none focus:border-indigo-500"
        />
        <button
          disabled={loading}
          className="rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 px-6 py-3 font-medium"
        >
          {loading ? "Searching…" : "Search"}
        </button>
      </form>

      <div className="max-w-2xl mx-auto flex flex-wrap gap-2 mb-10">
        <span className="text-sm text-neutral-500 self-center">Try:</span>
        {EXAMPLES.map((q) => (
          <button
            key={q}
            onClick={() => handleExample(q)}
            disabled={loading}
            className="rounded-full border border-neutral-700 bg-neutral-900 px-3 py-1 text-sm text-neutral-300 hover:border-indigo-500 hover:text-white disabled:opacity-50"
          >
            {q}
          </button>
        ))}
      </div>

      {error && <p className="text-center text-red-400 mb-6">{error}</p>}

      <div className="max-w-5xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-4">
        {results.map((r) => (
          <figure
            key={r.filename}
            className="rounded-lg overflow-hidden bg-neutral-900 border border-neutral-800"
          >
            <img src={r.url} alt="" className="w-full h-40 object-cover" />
            <figcaption className="px-3 py-2 text-sm text-neutral-400">score {r.score}</figcaption>
          </figure>
        ))}
      </div>
    </div>
  );
}

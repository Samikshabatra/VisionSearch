import { useState } from "react";
import { MOCK_RESULTS } from "./mockResults";

export default function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);

  function handleSearch(e) {
    e.preventDefault();
    // TODO(Week 7): POST query to FastAPI /search. For now, show mock results.
    setResults(MOCK_RESULTS);
  }

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100 px-6 py-10">
      <header className="max-w-3xl mx-auto text-center mb-8">
        <h1 className="text-4xl font-bold tracking-tight">VisionSearch</h1>
        <p className="text-neutral-400 mt-2">Type a sentence, get the matching images.</p>
      </header>

      <form onSubmit={handleSearch} className="max-w-2xl mx-auto flex gap-2 mb-10">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="two people on a beach at sunset"
          className="flex-1 rounded-lg bg-neutral-900 border border-neutral-700 px-4 py-3 outline-none focus:border-indigo-500"
        />
        <button className="rounded-lg bg-indigo-600 hover:bg-indigo-500 px-6 py-3 font-medium">
          Search
        </button>
      </form>

      <div className="max-w-5xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-4">
        {results.map((r) => (
          <figure
            key={r.id}
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

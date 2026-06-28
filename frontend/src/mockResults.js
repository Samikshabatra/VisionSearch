// Placeholder results until the FastAPI /search endpoint exists (Week 7).
export const MOCK_RESULTS = Array.from({ length: 8 }, (_, i) => ({
  id: i,
  url: `https://picsum.photos/seed/visionsearch${i}/400/300`,
  score: (0.92 - i * 0.04).toFixed(3),
}));

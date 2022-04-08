using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace Graph.Algorithms
{
    public record AsyncBreadthFirstSearch<T>(Predicate<T> Goal) : IAsyncAlgorithm<T>
        where T : IAsyncNode<T>
    {
        public IEnumerable<T> Discovered => Ordered;

        // Kludge: get around no "ordered set" in C#.
        private ConcurrentDictionary<T, bool> Visited { get; } = new();
        private ConcurrentQueue<T> Ordered { get; } = new();

        public async Task VisitAsync(T node)
        {
            var q = new ConcurrentQueue<T>();

            Discover(node, q);

            while (q.TryDequeue(out var v))
            {
                if (!Goal(v))
                {
                    await foreach (var w in v.TargetsAsync())
                    {
                        if (!Visited.ContainsKey(w))
                        {
                            Discover(w, q);
                        }
                    }
                }
            }
        }

        private void Discover(T node, ConcurrentQueue<T> queue)
        {
            lock (Visited)
                lock (Ordered)
                {
                    Visited[node] = true;
                    Ordered.Enqueue(node);
                    queue.Enqueue(node);
                }
        }
    }
}

using System;
using System.Collections.Generic;

namespace Graph.Algorithms
{
    /// <summary>
    /// Follows: https://en.wikipedia.org/wiki/Breadth-first_search .
    /// </summary>
    public record BreadthFirstSearch<T>(Predicate<T> Goal)
        : IAlgorithm<T>
        where T : INode<T>
    {
        public IEnumerable<T> Discovered => Ordered;

        // Kludge: get around no "ordered set" in C#.
        private HashSet<T> Visited { get; } = new();
        private List<T> Ordered { get; } = new();

        public void Visit(T node)
        {
            var q = new Queue<T>();

            Discover(node, q);

            while (q.TryDequeue(out var v))
            {
                if (!Goal(v))
                {
                    foreach (var w in v.Targets)
                    {
                        if (!Visited.Contains(w))
                        {
                            Discover(w, q);
                        }
                    }
                }
            }
        }

        private void Discover(T node, Queue<T> queue)
        {
            Visited.Add(node);
            Ordered.Add(node);

            queue.Enqueue(node);
        }
    }
}

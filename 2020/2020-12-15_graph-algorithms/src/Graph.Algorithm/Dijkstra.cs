using Graph.Core;
using System.Collections.Generic;

namespace Graph.Algorithm
{
    /// <summary>
    /// Implement Dijkstra's algorithm, following:
    /// https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm .
    /// </summary>
    public class Dijkstra<T>
        where T : INode<T>
    {
        public IReadOnlyDictionary<T, double> Visit(T node)
        {
            var result = new Dictionary<T, double>();

            // TODO: implementation.

            return result;
        }
    }
}

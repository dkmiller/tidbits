using System;
using System.Threading.Tasks;

namespace Graph.Algorithms
{
    public class AsyncBreadthFirstSearch<T> : IAsyncAlgorithm<T>
        where T : IAsyncNode<T>
    {
        public Task VisitAsync(T node)
        {
            throw new NotImplementedException();
        }
    }
}

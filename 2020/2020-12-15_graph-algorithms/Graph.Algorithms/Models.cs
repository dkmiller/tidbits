using System.Collections.Generic;
using System.Threading.Tasks;

namespace Graph.Algorithms
{
    public interface INode<out T>
       where T : INode<T>
    {
        IEnumerable<T> Targets { get; }
    }

    public interface IAsyncNode<out T>
        where T : IAsyncNode<T>
    {
        IAsyncEnumerable<T> TargetsAsync();
    }

    interface IAlgorithm<in T>
        where T : INode<T>
    {
        void Visit(T node);
    }

    interface IAsyncAlgorithm<in T>
        where T : IAsyncNode<T>
    {
        Task VisitAsync(T node);
    }
}

using System.Collections.Generic;

namespace Graph.Core
{
    public interface IAsyncNode<T>
    {
        IAsyncEnumerable<T> TargetsAsync();
    }
}

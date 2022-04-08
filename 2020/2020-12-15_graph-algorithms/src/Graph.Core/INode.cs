using System.Collections.Generic;

namespace Graph.Core
{
    public interface INode<out T>
    {
        public IEnumerable<T> Targets();
    }
}

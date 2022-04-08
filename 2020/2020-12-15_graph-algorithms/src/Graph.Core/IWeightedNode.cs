using System.Collections.Generic;

namespace Graph.Core
{
    interface IWeightedNode<out T>
    {
        public IEnumerable<IWeight<T>> Targets();
    }
}

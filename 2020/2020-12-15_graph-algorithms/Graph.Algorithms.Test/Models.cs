using System.Collections.Generic;

namespace Graph.Algorithms.Test
{
    record Node(string Label, params Node[] Out) : INode<Node>
    {
        public IEnumerable<Node> Targets => Out;
    }
}

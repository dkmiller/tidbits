using System.Collections.Generic;
using System.Threading.Tasks;

namespace Graph.Algorithms.Test
{
    record AsyncNode(string Label, params AsyncNode[] Out) : IAsyncNode<AsyncNode>
    {
        public async IAsyncEnumerable<AsyncNode> TargetsAsync()
        {
            foreach (var o in Out)
            {
                // https://gist.github.com/DCCoder90/d358ace7ef36401dd6f0449d4ab87706#gistcomment-3563710
                yield return await Task.FromResult(o);
            }
        }
    }

    record Node(string Label, params Node[] Out) : INode<Node>
    {
        public IEnumerable<Node> Targets => Out;
    }
}

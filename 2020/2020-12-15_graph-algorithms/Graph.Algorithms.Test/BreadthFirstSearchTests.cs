using Xunit;

namespace Graph.Algorithms.Test
{
    // TODO: make way of parsing strings to quickly create many examples?
    public class BreadthFirstSearchTests
    {
        [Fact]
        public void ReasonablyComplicatedGraph()
        {
            var h = new Node("h");
            var g = new Node("g");
            var f = new Node("f");
            var e = new Node("e", h);
            var d = new Node("d");
            var c = new Node("c", f, g);
            var b = new Node("b", d, e);
            var a = new Node("a", b, c);

            var bfs = new BreadthFirstSearch<Node>(n => false);
            bfs.Visit(a);

            Assert.Equal(new[] { a, b, c, d, e, f, g, h }, bfs.Discovered);
        }

        [Fact]
        public void StopsEarly()
        {
            var c = new Node("c");
            var b = new Node("b");
            var a = new Node("a", b, c);

            var bfs = new BreadthFirstSearch<Node>(n => n == b);
            bfs.Visit(a);

            Assert.Equal(new[] { a, b }, bfs.Discovered);
        }
    }
}

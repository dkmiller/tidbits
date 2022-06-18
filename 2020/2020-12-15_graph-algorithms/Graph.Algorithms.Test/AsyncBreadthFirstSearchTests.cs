using System.Threading;
using System.Threading.Tasks;
using Xunit;

namespace Graph.Algorithms.Test
{
    public class AsyncBreadthFirstSearchTests
    {
        [Fact]
        public async Task ReasonablyComplicatedGraph()
        {
            var h = new AsyncNode("h");
            var g = new AsyncNode("g");
            var f = new AsyncNode("f");
            var e = new AsyncNode("e", h);
            var d = new AsyncNode("d");
            var c = new AsyncNode("c", f, g);
            var b = new AsyncNode("b", d, e);
            var a = new AsyncNode("a", b, c);

            var bfs = new AsyncBreadthFirstSearch<AsyncNode>(n => false);
            await bfs.VisitAsync(a);

            Assert.Equal(new[] { a, b, c, d, e, f, g, h }, bfs.Discovered);
        }

        [Theory]
        [InlineData("https://en.wikipedia.org/wiki/Murray_Rothbard", "https://en.wikipedia.org/wiki/Labour_movement", 100)]
        [InlineData("https://en.wikipedia.org/wiki/Apocrypha", "https://en.wikipedia.org/wiki/Jesus", 100)]
        public async Task WikipediaSearch(string start, string end, int maxVisit)
        {
            var startNode = new Webpage(start, Webpage.Wikipedia);

            var visited = 0;
            var bfs = new AsyncBreadthFirstSearch<Webpage>(
                n => n.Url == end || Interlocked.Increment(ref visited) > maxVisit
            );
            await bfs.VisitAsync(startNode);

            Assert.Contains(bfs.Discovered, n => n.Url == end);
        }

        [Theory(Skip="TODO: fix this")]
        [InlineData("https://stackoverflow.com/q/62398572", "https://stackoverflow.com/q/2415742", 200)]
        public async Task StackoverflowSearch(string start, string end, int maxVisit)
        {
            var startNode = new Webpage(start, Webpage.Stackoverflow);

            var visited = 0;
            var bfs = new AsyncBreadthFirstSearch<Webpage>(
                n => n.Url == end || Interlocked.Increment(ref visited) > maxVisit
            );
            await bfs.VisitAsync(startNode);

            Assert.Contains(bfs.Discovered, n => n.Url == end);
        }
    }
}

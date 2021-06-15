using System.Linq;
using System.Threading.Tasks;
using Xunit;

namespace Graph.Algorithms.Test
{
    public class WikipediaTests
    {

        [Theory]
        [InlineData(
            "https://en.wikipedia.org/wiki/Breadth-first_search",
            "https://en.wikipedia.org/wiki/File:Question_book-new.svg")]
        public async Task WebpageContainsLinks(string url, params string[] expectedLinks)
        {
            var articles = await new Webpage(url, Webpage.Wikipedia)
                .TargetsAsync()
                .ToListAsync();

            Assert.All(expectedLinks,
                link => Assert.Contains(articles, w => w.Url == link)
                );
        }
    }
}

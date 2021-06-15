using System.Linq;
using System.Threading.Tasks;
using Xunit;

namespace Graph.Algorithms.Test
{
    public class WebpageTests
    {
        [Theory]
        [InlineData(
            "https://www.thomasclaudiushuber.com/2020/09/01/c-9-0-records-work-with-immutable-data-classes/",
            "https://www.thomasclaudiushuber.com/",
            "https://github.com/ababik/Remute")]
        [InlineData("https://stackoverflow.com/q/62398572",
            "https://stackoverflowbusiness.com/?ref=topbar_help",
            "https://docs.microsoft.com/en-us/dotnet/csharp/language-reference/configure-language-version")]
        public async Task WebpageContainsLinks(string url, params string[] expectedLinks)
        {
            var webpages = await new Webpage(url, Webpage.Default)
                .TargetsAsync()
                .ToListAsync();

            Assert.All(expectedLinks,
                link => Assert.Contains(webpages, w => w.Url == link)
                );
        }
    }
}

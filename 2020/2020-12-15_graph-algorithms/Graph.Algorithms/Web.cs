using HtmlAgilityPack;
using System.Collections.Generic;

namespace Graph.Algorithms
{
    // https://stackoverflow.com/a/20560385
    public delegate bool Keep(string v, out string u);

    public record Webpage(string Url, Keep Keep) : IAsyncNode<Webpage>
    {
        public async IAsyncEnumerable<Webpage> TargetsAsync()
        {
            var web = new HtmlWeb();
            var doc = await web.LoadFromWebAsync(Url);

            foreach (var node in doc.DocumentNode.SelectNodes("//a[@href]"))
            {
                // https://html-agility-pack.net/knowledge-base/47983842/how-to-get-a-url-from-the-href-attribute
                var value = node.Attributes["href"].Value;
                if (Keep(value, out var url))
                {
                    yield return new Webpage(url, Keep);
                }
            }
        }

        public static bool Default(string value, out string url)
        {
            url = value;
            return value.StartsWith("https://");
        }

        public static bool Stackoverflow(string value, out string url)
        {
            url = $"https://stackoverflow.com{value}";
            return value.StartsWith("/q/");
        }

        public static bool Wikipedia(string value, out string url)
        {
            url = $"https://en.wikipedia.org{value}";
            return value.StartsWith("/wiki");
        }
    }
}

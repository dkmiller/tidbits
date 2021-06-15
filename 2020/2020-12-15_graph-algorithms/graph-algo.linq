<Query Kind="Program">
  <NuGetReference>HtmlAgilityPack</NuGetReference>
  <NuGetReference>Microsoft.Extensions.Logging.Console</NuGetReference>
  <Namespace>HtmlAgilityPack</Namespace>
  <Namespace>System.Threading.Tasks</Namespace>
  <Namespace>Microsoft.Extensions.Logging</Namespace>
  <Namespace>System.Collections.Concurrent</Namespace>
</Query>

async Task Main()
{
	using var loggerFactory = LoggerFactory.Create(builder => builder.AddConsole());

	var wiki = new Wikipedia("https://en.wikipedia.org/wiki/Depth-first_search", loggerFactory.CreateLogger<Wikipedia>());

	var dfs = new AsyncBreadthFirstSearch<Wikipedia>(w => { w.ToString().Dump(); return Task.CompletedTask; });

	await dfs.GoAsync(wiki);
}

// You can define other methods, fields, classes and namespaces here

interface INode<out T>
	where T : INode<T>
{
	IEnumerable<T> Targets();
}

interface IAsyncNode<out T>
	where T : IAsyncNode<T>
{
	IAsyncEnumerable<T> TargetsAsync();
}

interface IAlgorithm<in T>
	where T : INode<T>
{
	void Go(INode<T> node);
}

interface IAsyncAlgorithm<in T>
	where T : IAsyncNode<T>
{
	Task GoAsync(IAsyncNode<T> node);
}

// https://en.wikipedia.org/wiki/Depth-first_search
record AsyncBreadthFirstSearch<T>(Func<IAsyncNode<T>, Task> VisitAsync)
	: IAsyncAlgorithm<T>
	where T : IAsyncNode<T>
{
	private ConcurrentDictionary<IAsyncNode<T>, bool> Visited { get; } = new();

	public async Task GoAsync(IAsyncNode<T> node)
	{
		if (Visited.TryAdd(node, true))
		{
			await VisitAsync(node);
			await foreach (var target in node.TargetsAsync())
			{
				await GoAsync(target);
			}
		}
	}
}

// TODO: make more general.
record Wikipedia(string Url, ILogger<Wikipedia> Logger) : IAsyncNode<Wikipedia>
{
	public async IAsyncEnumerable<Wikipedia> TargetsAsync()
	{
		// https://html-agility-pack.net/from-web
		Logger.LogInformation($"Downloading and parsing {Url}");
		var web = new HtmlWeb();
		var doc = await web.LoadFromWebAsync(Url);

		foreach (var node in doc.DocumentNode.SelectNodes("//a[@href]"))
		{
			// https://html-agility-pack.net/knowledge-base/47983842/how-to-get-a-url-from-the-href-attribute
			var value = node.Attributes["href"].Value;
			if (value.StartsWith("/wiki"))
			{
				yield return new Wikipedia($"https://en.wikipedia.org{value}", Logger);
			}
		}
	}
}
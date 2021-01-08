using Microsoft.Extensions.Logging;
using Microsoft.Graph;
using System.Threading.Tasks;

namespace Outlook.Calendar
{
    class Program
    {
        static async Task Main(string[] args)
        {
            using var loggerFactory = LoggerFactory.Create(builder => builder.AddConsole());
            var logger = loggerFactory.CreateLogger<Program>();

            var builder = new GraphClientBuilder(loggerFactory.CreateLogger<GraphClientBuilder>());
            // Get from https://developer.microsoft.com/en-us/graph/graph-explorer .
            var graphClient = await builder.BuildAsync("");

            var sro = new SearchRequestObject
            {
                EntityTypes = new[] { EntityType.Event },
                Query = new() { QueryString = "\"AML DS - feedback & support DRI\"" }
            };
            var page = await graphClient
                .Search
                .Query(new[] { sro })
                .Request()
                .PostAsync();

            foreach (var response in page)
            {
                foreach (var container in response.HitsContainers)
                {
                    foreach (var hit in container.Hits)
                    {
                        logger.LogInformation($"Found {hit.HitId}");

                        var @event = await graphClient
                            .Me
                            .Events[hit.HitId]
                            .Request()
                            .Select("subject,body,bodyPreview,organizer,attendees,start,end,location")
                            .GetAsync();

                        logger.LogInformation($"Detailed event: {@event}");
                    }
                }
            }
        }
    }
}

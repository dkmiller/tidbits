using Microsoft.Extensions.Logging;
using System.Threading.Tasks;

namespace Sharepoint.Upload
{
    class Program
    {
        async static Task Main(
            string accessToken = "",
            string hostname = "microsoft.sharepoint.com",
            string team = "AMLXLIAristotleworkinggroup",
            string filePath = @"\\FSU\Shares\TuringShare\",
            string filePattern = @"NLR_Models\Multilingual\TMLRv1\**",
            string targetPath = "General",
            int numFragments = 10)
        {
            // https://github.com/dotnet/runtime/issues/34742
            using var loggerFactory = LoggerFactory.Create(builder => builder.AddConsole());
            var logger = loggerFactory.CreateLogger<Program>();

            var builder = new GraphClientBuilder(logger);
            var graphClient = await builder.BuildAsync(accessToken);

            var filesClient = new FilesystemClient(filePath, filePattern);
            var files = filesClient.SearchAsync();

            var uploadClient = new UploadClient(hostname, team, targetPath, graphClient, logger, numFragments);

            await foreach (var file in files)
            {
                await uploadClient.UploadAsync(file);
            }

            logger.LogInformation("Done uploading files.");
        }
    }
}


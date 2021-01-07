using Microsoft.Extensions.Logging;
using System.Threading.Tasks;

namespace Sharepoint.Upload
{
    class Program
    {
        /// <summary>
        /// Upload files to SharePoint. For now, assumes directory structure already exists.
        /// </summary>
        /// <param name="hostname">Hostname for your SharePoint site.</param>
        /// <param name="team">SharePoint team.</param>
        /// <param name="root">Directory to search for files to upload.</param>
        /// <param name="glob">Pattern to search inside filePath.</param>
        /// <param name="target">Target directory inside the SharePoint site.</param>
        /// <param name="token">
        /// (Optional) AAD access token for auth.
        /// You may obtain this by visiting: https://developer.microsoft.com/en-us/graph/graph-explorer .
        /// </param>
        /// <param name="numFragments">Multiple of 320 KiB to include in each file fragment.</param>
        async static Task Main(
            string hostname,
            string team,
            string root,
            string glob,
            string target,
            string token = "",
            int numFragments = 10)
        {
            // https://github.com/dotnet/runtime/issues/34742
            using var loggerFactory = LoggerFactory.Create(builder => builder.AddConsole());
            var logger = loggerFactory.CreateLogger<Program>();

            var filesClient = new FilesystemClient(root, glob);
            var files = filesClient.SearchAsync();

            var builder = new GraphClientBuilder(logger);
            var graphClient = await builder.BuildAsync(token);

            var uploadClient = new UploadClient(hostname, team, target, numFragments, graphClient, loggerFactory);

            await foreach (var file in files)
            {
                await uploadClient.UploadAsync(file);
            }

            logger.LogInformation("Done uploading files.");
        }
    }
}


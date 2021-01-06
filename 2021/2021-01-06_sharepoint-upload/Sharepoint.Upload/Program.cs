using Microsoft.Extensions.Logging;
using Microsoft.Graph;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
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
            string filePattern = @"NLR_Models\Monolingual\NLRv1-Base-Uncased\PT\**",
            string targetPath = "General")
        {
            // https://github.com/dotnet/runtime/issues/34742
            using var loggerFactory = LoggerFactory.Create(builder => builder.AddConsole());
            var logger = loggerFactory.CreateLogger<Program>();

            var builder = new GraphClientBuilder(logger);
            var graphClient = await builder.BuildAsync(accessToken);

            var site = await graphClient
                .Sites
                .GetByPath($"/teams/{team}", hostname)
                .Request()
                .GetAsync();

            var drive = await graphClient
                .Sites[site.Id]
                .Drive
                .Request()
                .GetAsync();

            var paths = Paths(filePath, filePattern);

            foreach (var path in paths)
            {
                var dir = Path.GetDirectoryName(path).Replace(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
                var file = Path.GetFileName(path);
                var fi = new FileInfo($"{filePath}/{path}");
                var fileSize = fi.Length;

                var driveItem = await graphClient
                    .Drives[drive.Id]
                    .Root
                    .ItemWithPath($"{ targetPath}/{ dir}")
                    .Request()
                    .GetAsync();

                //driveItem.ParentReference.id

                var props = new DriveItemUploadableProperties { Name = file };//, FileSize = fileSize };
                props.FileSystemInfo = new Microsoft.Graph.FileSystemInfo
                {
                    CreatedDateTime = fi.CreationTimeUtc,
                    LastAccessedDateTime = fi.LastAccessTimeUtc,
                    LastModifiedDateTime = fi.LastWriteTimeUtc,
                };

                try
                {

                    //var uploadSession = await graphClient
                    //    .Drive
                    //    .Items["01KGPRHTV6Y2GOVW7725BZO354PWSELRRZ"].ItemWithPath("_hamilton.png").CreateUploadSession().Request().PostAsync();


                    var req = graphClient
                        .Drives[drive.Id]
                        .Items[driveItem.Id]
                        .ItemWithPath(file)
                        .CreateUploadSession()
                        .Request();

                    var url = req.RequestUrl;
                    var body = req.RequestBody;

                    var __ = await graphClient
                        .Drives[drive.Id]
                        .Items[driveItem.Id]
                        .Request()
                        .GetAsync();


                    var session = await req
                        .PostAsync();

                    using var uploadStream = System.IO.File.OpenRead(fi.FullName);

                    var upload = new LargeFileUploadTask<DriveItem>(session, uploadStream, maxSliceSize: 320 * 1024);

                    var uploadedFiles = await upload.ResumeAsync(new Progress(logger, fileSize));
                    //var uploaded = 

                    logger.LogInformation($"{uploadedFiles}");
                    //var cancelRes = await new HttpClient().DeleteAsync(session.UploadUrl);

                    //Console.WriteLine(cancelRes);

                }
                catch (ServiceException e)
                {
                    Console.WriteLine(e);
                }

                //var folder = 
                Console.WriteLine(file);
            }
        }

        private static IEnumerable<string> Paths(string path, string pattern) =>
            System.IO.Directory.GetFiles(path, pattern, SearchOption.AllDirectories)
            .Select(
                file => file.Replace(path, "").Replace(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)
                );

        record Progress(ILogger Logger, long Total) : IProgress<long>
        {
            public void Report(long value)
            {
                var percentage = (value * 100) / Total;

                Logger.LogInformation($"Upload in progress: {value} bytes of {Total} ({percentage} percent).");
            }
        }
    }
}

using Microsoft.Extensions.Logging;
using Microsoft.Graph;
using System.Threading.Tasks;

namespace Sharepoint.Upload
{
    record UploadClient(string Hostname, string Team, string Target, int NumFragments, IGraphServiceClient GraphClient, ILoggerFactory Factory)
    {
        private ILogger Logger { get; } = Factory.CreateLogger<UploadClient>();
            
        // See: https://docs.microsoft.com/en-us/graph/api/driveitem-createuploadsession?view=graph-rest-1.0#upload-bytes-to-the-upload-session
        const int MinFragmentSize = 320 * 1024;

        public async Task UploadAsync(File file)
        {
            Logger.LogInformation($"Uploading {file}");
            var drive = await DriveAsync();
            var path = $"{Target}/{file.Directory}";

            var driveItem = await GraphClient
                .Drives[drive.Id]
                .Root
                .ItemWithPath(path)
                .Request()
                .GetAsync();

            var session = await GraphClient
                .Drives[drive.Id]
                .Items[driveItem.Id]
                .ItemWithPath(file.Name)
                .CreateUploadSession()
                .Request()
                .PostAsync();

            using var uploadStream = System.IO.File.OpenRead(file.Info.FullName);

            var upload = new LargeFileUploadTask<DriveItem>(session, uploadStream, maxSliceSize: NumFragments * MinFragmentSize);

            var progress = new FileProgress(file.Name, file.Info.Length, Factory.CreateLogger<FileProgress>());
            var uploadResult = await upload.ResumeAsync(progress);

            Logger.LogInformation($"Finished uploading {file.Name}. Status: {uploadResult.UploadSucceeded}");
        }

        private async Task<Drive> DriveAsync()
        {
            var site = await GraphClient
                .Sites
                .GetByPath($"/teams/{Team}", Hostname)
                .Request()
                .GetAsync();

            var drive = await GraphClient
                .Sites[site.Id]
                .Drive
                .Request()
                .GetAsync();

            return drive;
        }
    }
}

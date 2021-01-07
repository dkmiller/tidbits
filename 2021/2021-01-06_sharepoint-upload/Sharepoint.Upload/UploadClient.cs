using Microsoft.Extensions.Logging;
using Microsoft.Graph;
using System.Net;
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

            var driveItem = await GetOrCreateDriveItemAsync(drive, path);
            var itemRequestBuilder = GraphClient
                .Drives[drive.Id]
                .Items[driveItem.Id]
                .ItemWithPath(file.Name);

            await ConfirmExistsRemotelyAsync(itemRequestBuilder, file);
        }

        private async Task ConfirmExistsRemotelyAsync(IDriveItemRequestBuilder requestBuilder, File file)
        {
            try
            {
                var fileItem = await requestBuilder
                .Request()
                .GetAsync();

                if (fileItem.Size == file.Info.Length)
                {
                    Logger.LogInformation($"Remote version already exists with same size.");
                }
                else
                {
                    Logger.LogWarning($"Remote version with different size exists (remote: {fileItem.Size}, local: {file.Info.Length})!");
                }
            }
            catch (ServiceException e) when (e.StatusCode == HttpStatusCode.NotFound)
            {
                Logger.LogInformation($"Remote version does not exist, uploading file.");
                var session = await requestBuilder
                    .CreateUploadSession()
                    .Request()
                    .PostAsync();

                using var uploadStream = System.IO.File.OpenRead(file.Info.FullName);

                var upload = new LargeFileUploadTask<DriveItem>(session, uploadStream, maxSliceSize: NumFragments * MinFragmentSize);

                var progress = new FileProgress(file.Name, file.Info.Length, Factory.CreateLogger<FileProgress>());
                var uploadResult = await upload.ResumeAsync(progress);

                Logger.LogInformation($"Finished uploading {file.Name}. Status: {uploadResult.UploadSucceeded}");
            }
        }

        private async Task<DriveItem> GetOrCreateDriveItemAsync(Drive drive, string path)
        {
            Logger.LogInformation($"Finding drive with path {path}");

            DriveItem result;

            try
            {
                result = await GraphClient
                    .Drives[drive.Id]
                    .Root
                    .ItemWithPath(path)
                    .Request()
                    .GetAsync();
            }
            catch (ServiceException e) when (e.StatusCode == HttpStatusCode.NotFound)
            {
                // TODO: this doesn't work! Follow:
                // https://docs.microsoft.com/en-us/graph/api/driveitem-post-children
                Logger.LogWarning($"Drive item with path {path} does not exist; creating it.");
                var item = new DriveItem();
                result = await GraphClient
                    .Drives[drive.Id]
                    .Root
                    .ItemWithPath(path)
                    .Request()
                    .CreateAsync(item);
            }

            return result;
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

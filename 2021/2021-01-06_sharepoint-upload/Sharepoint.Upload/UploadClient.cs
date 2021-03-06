﻿using ByteSizeLib;
using Microsoft.Extensions.Logging;
using Microsoft.Graph;
using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Text.RegularExpressions;
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
            var drive = await GetDriveAsync();
            Logger.LogInformation($"Using drive with ID {drive.Id}");
            var path = $"{Target}/{file.Directory}";

            var driveItem = await GetOrCreateFolderAsync(drive, path);
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
                var size = ByteSize.FromBytes(file.Info.Length);
                Logger.LogInformation($"Remote version does not exist, uploading {size} file.");
                var session = await requestBuilder
                    .CreateUploadSession()
                    .Request()
                    .PostAsync();

                using var uploadStream = System.IO.File.OpenRead(file.Info.FullName);

                var upload = new LargeFileUploadTask<DriveItem>(session, uploadStream, maxSliceSize: NumFragments * MinFragmentSize);

                var progress = new FileProgress(file.Name, size, Factory.CreateLogger<FileProgress>());
                var uploadResult = await upload.ResumeAsync(progress);

                Logger.LogInformation($"Finished uploading {file.Name}. Status: {uploadResult.UploadSucceeded}");
            }
        }

        private async Task<DriveItem> GetOrCreateFolderAsync(Drive drive, string path)
        {
            Logger.LogInformation($"Finding drive item with path {path}");

            DriveItem result;

            try
            {
                var requestBuilder = GraphClient
                    .Drives[drive.Id]
                    .Root;

                if (!Regex.IsMatch(path, "^/+$|^$"))
                {
                    requestBuilder = requestBuilder.ItemWithPath(path);
                }

                result = await requestBuilder
                    .Request()
                    .GetAsync();

                Logger.LogInformation($"Drive item already exists with ID {result.Id}");
            }
            catch (ServiceException e) when (e.StatusCode == HttpStatusCode.NotFound)
            {
                Logger.LogWarning($"Drive item with path {path} does not exist; creating it.");
                var segments = path.Split(Path.AltDirectorySeparatorChar, options: StringSplitOptions.RemoveEmptyEntries);
                var parentSegments = segments[..^1];
                var parentPath = string.Join(Path.AltDirectorySeparatorChar, parentSegments);
                var parent = await GetOrCreateFolderAsync(drive, parentPath);

                // https://docs.microsoft.com/en-us/graph/api/driveitem-post-children
                var newDrive = new DriveItem
                {
                    Name = segments[^1],
                    // Don't remove this. It's needed to tell SharePoint that we're creating a folder.
                    Folder = new(),
                    AdditionalData = new Dictionary<string, object>
                    {
                        ["@microsoft.graph.conflictBehavior"] = "replace"
                    }
                };

                result = await GraphClient
                    .Drives[drive.Id]
                    .Items[parent.Id]
                    .Children
                    .Request()
                    .AddAsync(newDrive);

                Logger.LogInformation($"Created drive item with ID {drive.Id}");
            }

            return result;
        }

        private async Task<Drive> GetDriveAsync()
        {
            var site = await GraphClient
                .Sites
                .GetByPath($"/teams/{Team}", Hostname)
                .Request()
                .GetAsync();
            Logger.LogInformation($"Using site with ID {site.Id}");

            var drive = await GraphClient
                .Sites[site.Id]
                .Drive
                .Request()
                .GetAsync();

            return drive;
        }
    }
}

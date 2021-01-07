using ByteSizeLib;
using Microsoft.Extensions.Logging;
using System;

namespace Sharepoint.Upload
{
    record FileProgress(string Name, ByteSize Total, ILogger Logger) : IProgress<long>
    {
        public void Report(long value)
        {
            var totalBytes = Total.Bytes;
            var percentage = value * 100 / totalBytes;
            var valueSize = ByteSize.FromBytes(value);

            Logger.LogInformation($"Uploading {Name}: {percentage}% = {valueSize} / {Total}.");
        }
    }
}

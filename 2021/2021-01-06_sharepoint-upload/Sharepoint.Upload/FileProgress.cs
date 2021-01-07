using ByteSizeLib;
using Microsoft.Extensions.Logging;
using System;

namespace Sharepoint.Upload
{
    record FileProgress(string Name, long Total, ILogger Logger) : IProgress<long>
    {
        private ByteSize TotalSize { get; } = ByteSize.FromBytes(Total);

        public void Report(long value)
        {
            var percentage = (value * 100) / Total;
            var valueSize = ByteSize.FromBytes(value);

            Logger.LogInformation($"Uploading {Name} in progress: {valueSize} of {TotalSize} ({percentage} percent).");
        }
    }
}

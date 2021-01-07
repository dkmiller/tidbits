using Microsoft.Extensions.Logging;
using System;

namespace Sharepoint.Upload
{
    record Progress(ILogger Logger, long Total) : IProgress<long>
    {
        public void Report(long value)
        {
            var percentage = (value * 100) / Total;

            Logger.LogInformation($"Upload in progress: {value} bytes of {Total} ({percentage} percent).");
        }
    }
}

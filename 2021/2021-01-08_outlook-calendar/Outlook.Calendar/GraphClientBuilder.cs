using Azure.Core;
using Azure.Identity;
using Microsoft.Extensions.Logging;
using Microsoft.Graph;
using System.Net.Http.Headers;
using System.Threading.Tasks;

namespace Outlook.Calendar
{
    record GraphClientBuilder(ILogger Logger)
    {
        private DefaultAzureCredential Credential { get; } = new();

        private async Task<string> GetAccessTokenAsync()
        {
            Logger.LogInformation("Obtaining access token using Azure credential.");
            var token = await Credential.GetTokenAsync(
                // TODO: getting unauthorized exception.
                new TokenRequestContext(new[] { "https://graph.microsoft.com/Calendars.ReadWrite" })
                );

            var accessToken = token.Token;
            Logger.LogInformation($"Obtained access token:\n{accessToken}\n");
            return accessToken;
        }

        public async Task<IGraphServiceClient> BuildAsync(string accessToken = "")
        {
            if (string.IsNullOrWhiteSpace(accessToken))
            {
                accessToken = await GetAccessTokenAsync();
            }

            var graphClient = new GraphServiceClient(new DelegateAuthenticationProvider(m =>
            {
                m.Headers.Authorization = new AuthenticationHeaderValue("Bearer", accessToken);
                return Task.FromResult(0);
            }));

            return graphClient;
        }
    }
}

using Microsoft.Extensions.Logging;
using Microsoft.Graph;
using Microsoft.Identity.Client;
using System.Net.Http.Headers;
using System.Threading.Tasks;

namespace Sharepoint.Upload
{
    record GraphClientBuilder(
        ILogger Logger,
        string ClientId = "365ae7bd-e462-4dbe-9625-ac6e71e5ea03",
        string RedirectUri = "http://localhost",
        string TenantId = "72f988bf-86f1-41af-91ab-2d7cd011db47"
        )
    {
        private IPublicClientApplication App { get; } = PublicClientApplicationBuilder
            .Create(ClientId)
            .WithRedirectUri(RedirectUri)
            .WithAuthority(AzureCloudInstance.AzurePublic, TenantId)
            .Build();

        private async Task<string> GetAccessTokenAsync()
        {
            Logger.LogInformation("Obtaining access token using interactive authentication.");
            var authResult = await App
                .AcquireTokenInteractive(new[] { "https://graph.microsoft.com/.default" })
                .ExecuteAsync();

            var accessToken = authResult.AccessToken;
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

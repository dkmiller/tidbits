using Azure.Core;
using Azure.Identity;
using Microsoft.Extensions.Logging;
using Microsoft.Graph;
using System.Linq;
using System.Net.Http.Headers;
using System.Threading.Tasks;

namespace Outlook.Calendar
{
    record GraphClientBuilder(ILogger Logger)
    {
        private DefaultAzureCredential Credential { get; } = new();

        private const string Scopes = "Calendars.ReadWrite Contacts.ReadWrite DeviceManagementApps.ReadWrite.All DeviceManagementConfiguration.Read.All DeviceManagementConfiguration.ReadWrite.All DeviceManagementManagedDevices.PrivilegedOperations.All DeviceManagementManagedDevices.Read.All DeviceManagementManagedDevices.ReadWrite.All DeviceManagementRBAC.Read.All DeviceManagementRBAC.ReadWrite.All DeviceManagementServiceConfig.Read.All DeviceManagementServiceConfig.ReadWrite.All Directory.AccessAsUser.All Directory.ReadWrite.All Files.ReadWrite.All Group.ReadWrite.All IdentityRiskEvent.Read.All Mail.ReadWrite MailboxSettings.ReadWrite Notes.ReadWrite.All openid People.Read Presence.Read Presence.Read.All profile Reports.Read.All Sites.ReadWrite.All Tasks.ReadWrite User.Read User.ReadBasic.All User.ReadWrite User.ReadWrite.All email";

        private async Task<string> GetAccessTokenAsync()
        {
            Logger.LogInformation("Obtaining access token using Azure credential.");

            // TODO: getting unauthorized exception.
            // Not all scopes are respected :( .
            // Get this list by pasting access token into https://jwt.ms and looking under `scp`.
            var scopes = Scopes.Split(" ")
                .Select(scope => $"https://graph.microsoft.com/{scope}")
                .ToArray();
            //var scopes = new[]
            //{
            //    "https://graph.microsoft.com/Calendars.ReadWrite",
            //    "https://graph.microsoft.com/Contacts.ReadWrite",
            //    "https://graph.microsoft.com/User.ReadWrite.All",
            //};
            var token = await Credential.GetTokenAsync(
                new TokenRequestContext(scopes)
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

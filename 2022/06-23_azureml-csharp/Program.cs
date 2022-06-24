// See https://aka.ms/new-console-template for more information

using Azure;
using Azure.Core;
using Azure.Identity;
using Azure.ResourceManager;
using Azure.ResourceManager.MachineLearning;
using Azure.ResourceManager.MachineLearning.Models;

class Program
{
    static async Task Main(
        string subscriptionId = "48bbc269-ce89-4f6f-9a12-c6f91fcb772d",
        string resourceGroupName = "aml1p-rg",
        string workspaceName = "aml1p-ml-wus2",
        string clusterName = "cpucluster"
        )
    {
        // https://github.com/Azure/azure-sdk-for-net/blob/main/doc/dev/mgmt_quickstart.md
        var credential = new DefaultAzureCredential();
        var armClient = new ArmClient(credential);
        var subscription = armClient.GetSubscriptionResource(new ResourceIdentifier($"/subscriptions/{subscriptionId}"));
        var resourceGroup = await subscription.GetResourceGroupAsync(resourceGroupName);
        var workspace = await resourceGroup.Value.GetMachineLearningWorkspaceAsync(workspaceName);

        var mlFlow = workspace.Value.Data.MlFlowTrackingUri;
        Console.WriteLine($"MLFlow tracking URI: {mlFlow}");

        var jobs = workspace.Value.GetMachineLearningJobs();

        foreach (var job in jobs.Take(2))
        {
            var props = job.Data.Properties;
            Console.WriteLine($"{props.DisplayName} ({props.Status}) == {props.Description}");
        }

        // https://github.com/Azure/azureml-samples-dotnet/blob/main/Program.cs
        var commandJob = new CommandJob("printenv", "azureml://registries/CuratedRegistry/environments/AzureML-sklearn-1.0-ubuntu20.04-py38-cpu/versions/23")
        {
            ComputeId = $"{workspace.Value.Id}/computes/{clusterName}",
            ExperimentName = "danmill-csharp"
        };
        var jobData = new MachineLearningJobData(commandJob);
        var jobOperation = await jobs.CreateOrUpdateAsync(WaitUntil.Completed, Guid.NewGuid().ToString(), jobData);
        var jobResource = jobOperation.Value;
        Console.WriteLine($"Created job {jobResource.Data.Id}.");
    }
}

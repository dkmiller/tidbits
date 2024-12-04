// TODO: https://learn.microsoft.com/en-us/dotnet/orleans/deployment/kubernetes
// https://learn.microsoft.com/en-us/dotnet/orleans/tutorials-and-samples/tutorial-1


using Orleans;
using Orleans.Hosting;
using OrleansHttp.Grains;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using HelloWorld;

// Configure the host
using var host = new HostBuilder()
    .UseOrleans(builder => builder.UseLocalhostClustering())
    .Build();

// Start the host
await host.StartAsync();

// Get the grain factory
var grainFactory = host.Services.GetRequiredService<IGrainFactory>();

// Get a reference to the HelloGrain grain with the key "friend"
var friend = grainFactory.GetGrain<IHello>("friend");

// Call the grain and print the result to the console
var result = await friend.SayHello("Good morning!");
System.Console.WriteLine($"""

    {result}

    """);

System.Console.WriteLine("Orleans is running.\nPress Enter to terminate...");
System.Console.ReadLine();
System.Console.WriteLine("Orleans is stopping...");

await host.StopAsync();

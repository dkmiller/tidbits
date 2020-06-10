using System;
using Microsoft.Extensions.Logging;
using Orleans;
using Orleans.Configuration;
using Orleans.Hosting;
using System.Threading.Tasks;

namespace dkmiller.Silo
{
    class Program
    {
        public static async Task Main(string[] args)
        {
            var host = new SiloHostBuilder()
                .UseLocalhostClustering()
                .Configure<ClusterOptions>(options =>
                {
                    options.ClusterId = "dev";
                    options.ServiceId = "OrleansBasics";
                })
                .ConfigureApplicationParts(
                    parts => parts.AddApplicationPart(typeof(Arithmetic).Assembly)
                                .WithReferences()
                    )
                .ConfigureLogging(log => log.AddConsole())
                .Build();

                await host.StartAsync();
                Console.WriteLine("Press Enter to terminate...\n\n");
                Console.ReadLine();

                await host.StopAsync();
        }
    }
}

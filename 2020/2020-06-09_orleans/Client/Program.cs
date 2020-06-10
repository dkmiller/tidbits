using dkmiller.Core;
using Microsoft.Extensions.Logging;
using Orleans;
using Orleans.Configuration;
using System;
using System.Threading.Tasks;

namespace dkmiller.Client
{
    class Program
    {
        static async Task Main(string[] args)
        {
            using var client = new ClientBuilder()
                .UseLocalhostClustering()
                .Configure<ClusterOptions>(options =>
                {
                    options.ClusterId = "dev";
                    options.ServiceId = "OrleansBasics";
                })
                .ConfigureLogging(logging => logging.AddConsole())
                .Build();

                await client.Connect();
                Console.WriteLine("Connected to host.");

                await Work(client);

                Console.WriteLine("Type ENTER to exit...");
                Console.ReadKey();
        }

        private static async Task Work(IClusterClient client)
        {
            var friend = client.GetGrain<IArithmetic>(new Guid());
            var response = await friend.NextPrime(10000);
            Console.WriteLine($"\n\n{response}\n\n");
        }
    }
}

using Microsoft.AspNetCore.Hosting;
using System;

namespace kestral
{
    class Program
    {
        static void Main(string[] args) =>
            new WebHostBuilder()
                .UseKestrel()
                .UseStartup<Startup>()
                .Build()
                .Run();
    }
}

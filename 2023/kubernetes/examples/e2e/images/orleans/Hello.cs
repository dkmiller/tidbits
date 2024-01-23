using Orleans;
using System.Threading.Tasks;

namespace HelloWorld;

public sealed class HelloGrain : Grain, IHello
{
    public ValueTask<string> SayHello(string greeting) =>
        ValueTask.FromResult($"Hello, {greeting}!");
}
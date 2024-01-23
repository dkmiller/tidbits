using Orleans;
using System.Threading.Tasks;

namespace HelloWorld;

public interface IHello : IGrainWithStringKey
{
    ValueTask<string> SayHello(string greeting);
}

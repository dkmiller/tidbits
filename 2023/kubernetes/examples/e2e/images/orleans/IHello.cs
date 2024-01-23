using Orleans;
using System.Threading.Tasks;

namespace GrainInterfaces;

public interface IHello : IGrainWithStringKey
{
    ValueTask<string> SayHello(string greeting);
}

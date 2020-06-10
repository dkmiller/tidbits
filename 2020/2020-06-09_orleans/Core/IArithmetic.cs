using Orleans;
using System.Threading.Tasks;

namespace dkmiller.Core
{
    public interface IArithmetic : IGrainWithGuidKey
    {
        Task<long> NextPrime(long index);
    }
}

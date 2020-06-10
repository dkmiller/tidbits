using Orleans;
using dkmiller.Core;
using Microsoft.Extensions.Logging;
using Open.Numeric.Primes.Extensions;
using System.Threading.Tasks;


namespace dkmiller.Silo
{
    public class Arithmetic : Grain, IArithmetic
    {
        private ILogger Logger { get; }

        public Arithmetic(ILogger<Arithmetic> logger) =>
            Logger = logger;
        
        public Task<long> NextPrime(long index)
        {
            Logger.LogInformation($"Calculating {nameof(NextPrime)}({index}).");
            var result = index.NextPrime();
            Logger.LogInformation($"The next prime is {result}.");
            return Task.FromResult(result);
        }
    }
}

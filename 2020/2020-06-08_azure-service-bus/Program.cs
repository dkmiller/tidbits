using Microsoft.Azure.ServiceBus;
using System;
using System.Text;
using System.Threading.Tasks;

namespace dkmiller.AzureServiceBus.Client
{
    class Program
    {
        /// <summary>
        /// Entry point.
        /// </summary>
        /// <param name="connectionString">Azure Service Bus connection string.</param>
        /// <param name="maxConcurrentCalls">Maximum number of concurrent message receives.</param>
        /// <param name="numMessagesSend">Send this many messages.</param>
        /// <param name="queueName"> Azure Service Bus queue name.</param>
        static async Task Main(
            string connectionString,
            string queueName = "test_queue",
            int numMessagesSend = 10,
            int maxConcurrentCalls = 2)
        {
            Console.WriteLine("Hello World!");
            var client = new QueueClient(connectionString, queueName);

            var messageHandlerOptions = new MessageHandlerOptions(e =>
            {
                Console.WriteLine(e);
                return Task.CompletedTask;
            })
            {
                MaxConcurrentCalls = maxConcurrentCalls,
                AutoComplete = false
            };

            client.RegisterMessageHandler(async (message, _) =>
            {
                Console.WriteLine($"Received message (seq # {message.SystemProperties.SequenceNumber}): '{Encoding.Default.GetString(message.Body)}'.");
                await client.CompleteAsync(message.SystemProperties.LockToken);
            },
            messageHandlerOptions);

            await SendMessages(numMessagesSend, client);

            Console.WriteLine("Enter key to finish.");
            Console.ReadKey();

            await client.CloseAsync();
        }

        private static async Task SendMessages(int numMessages, IQueueClient client)
        {
            for (var i = 0; i < numMessages; ++i)
            {
                var body = $"Message {i}";
                var message = new Message(Encoding.Default.GetBytes(body));
                Console.WriteLine($"Sending message: '{body}'.");
                await client.SendAsync(message);
            }
        }
    }
}

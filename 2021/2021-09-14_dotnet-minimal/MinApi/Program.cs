var builder = WebApplication.CreateBuilder(args);

builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen(c =>
{
    c.SwaggerDoc("v1", new() { Title = builder.Environment.ApplicationName, Version = "v1" });
});

var app = builder.Build();

// Probably only want to do in development environment?
app.UseSwagger();
app.UseSwaggerUI(c => c.SwaggerEndpoint("/swagger/v1/swagger.json", $"{builder.Environment.ApplicationName} v1"));

app.MapGet("/hello", () => "Hello, World!");
app.MapGet("/hello/name/{firstName}/{lastName}", (string firstName, string lastName) => $"Hello {firstName} {lastName}");

// Handle request strongly-typed body.
app.MapPost("/hello", (Person person) => $"Hello, {person.Name} ({person.Age})");

app.MapGet("/age/{age:range(18, 120)}", (int age) => $"You're {age} years old!");
app.MapGet("/user/{id:guid}", (Guid id) => $"User {id.ToString("X")}");

app.MapGet("/isadult", (int age) => new { IsAdult = age >= 18 });

var logger = app.Logger;
var config = app.Configuration.GetSection("Config").Get<Config>();
logger.LogInformation($"Hi from logging (foo = {config.Foo}!");

app.Run();

record Person(string Name, int Age);
record Config { public string Foo { get; init; } = "default"; }

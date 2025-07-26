# https://learn.microsoft.com/en-us/aspnet/core/host-and-deploy/docker/building-net-docker-images?view=aspnetcore-8.0
# https://www.docker.com/blog/9-tips-for-containerizing-your-net-application/
# https://github.com/dotnet/dotnet-docker/blob/main/samples/dotnetapp/Dockerfile

# https://github.com/OrleansContrib/Orleans.Http/blob/master/Samples/OrleansHttp.Grains/IHelloGrain.cs

FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src

COPY *.csproj .
RUN dotnet restore

COPY . .
RUN dotnet publish --no-restore -o /app

FROM mcr.microsoft.com/dotnet/runtime:8.0
WORKDIR /app
COPY --from=build /app .

ENTRYPOINT ["./orleans-api"]

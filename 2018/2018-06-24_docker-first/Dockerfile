# Official .NET Core image.
FROM microsoft/dotnet:2.1-sdk

WORKDIR /app
ADD . /app

RUN dotnet build
CMD ["dotnet", "run", "random text"]

# Super small Python image.
FROM stefanscherer/python-windows:nano

WORKDIR /app
ADD . /app

RUN pip install --trusted-host pypi.python.org -r requirements.txt
ENV NAME World
CMD ["python", "app.py"]

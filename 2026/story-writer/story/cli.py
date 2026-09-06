from cyclopts import App

app = App()


@app.command
def write(chapters: int = 10):
    for i in range(chapters):
        print(f"Looping! {i}")


app()

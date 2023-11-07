docker build -t pyc .

# https://stackoverflow.com/a/37340588/
docker run --volume $PWD:/run pyc --distpath /run

# https://unix.stackexchange.com/a/185039
echo "Generated $(stat -f%z cli) bytes"

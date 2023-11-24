docker build -t ssh .

# Cleanup any pre-existing port stuff
# Get only the port from lsof: https://stackoverflow.com/a/62453482/
kill -9 $(lsof -t -i :24464)
kill -9 $(lsof -t -i :63752)

# Clean up any pre-existing SSH configuration.
# sed -i '' "/localhost/d" ~/.ssh/known_hosts

docker run -e PUBLIC_KEY="$(cat ~/.ssh/id_rsa_16505ade1dbd42f38623fd2aef236a27.pub)" -e USER_NAME=dan -p 2222:2222 -p 24464:24464 ssh

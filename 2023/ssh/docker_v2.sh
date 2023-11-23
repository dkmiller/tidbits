docker build -t ssh .

sed -i '' "/localhost/d" ~/.ssh/known_hosts

docker run -e PUBLIC_KEY="$(cat ~/.ssh/id_rsa_16505ade1dbd42f38623fd2aef236a27.pub)" -e USER_NAME=dan -p 2222:2222 -p 24464:24464 ssh

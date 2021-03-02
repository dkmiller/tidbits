# https://marriageheat.com/2021/02/27/no-question-about-my-love/

Invoke-WebRequest -Uri "https://marriageheat.com/wp-admin/admin-ajax.php" `
-Method "POST" `
-Headers @{
"method"="POST"
  "authority"="marriageheat.com"
  "scheme"="https"
  "path"="/wp-admin/admin-ajax.php"
  "accept"="application/json, text/javascript, */*; q=0.01"
  "dnt"="1"
  "x-requested-with"="XMLHttpRequest"
  "user-agent"="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.182 Safari/537.36 Edg/88.0.705.81"
  "origin"="https://marriageheat.com"
  "sec-fetch-site"="same-origin"
  "sec-fetch-mode"="cors"
  "sec-fetch-dest"="empty"
  "referer"="https://marriageheat.com/2021/02/27/no-question-about-my-love/"
  "accept-encoding"="gzip, deflate, br"
  "accept-language"="en-US,en;q=0.9"
} `
-ContentType "application/x-www-form-urlencoded; charset=UTF-8" `
-Body "action=load_results&postID=133918&nonce=41908004f0"

# {"voteCount":35,"avgRating":4.9000000000000004,"errorMsg":""}

# Works!

iwr https://marriageheat.com/wp-admin/admin-ajax.php -method Post -Body "action=load_results&postID=133918&nonce=41908004f0"
# Returns:
# {"voteCount":35,"avgRating":4.9000000000000004,"errorMsg":""}

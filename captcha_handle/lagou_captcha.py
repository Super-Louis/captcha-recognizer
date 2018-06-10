import requests
res = requests.get('https://passport.lagou.com/vcode/create?from=register&amp;refresh=4').content
with open('lagou.jpg', 'wb') as f:
    f.write(res)
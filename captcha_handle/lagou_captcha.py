import requests
headers = {
    'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.75 Safari/537.36'
}
def get_captcha():
    res = requests.get('https://passport.lagou.com/vcode/create?from=register&amp;refresh=1').content
    with open('lagou.jpg', 'wb') as f:
        f.write(res)
if __name__ == '__main__':
    get_captcha()
# from PIL import Image as img
# import numpy as np
# i = img.open('lagou.jpg')
# i = i.convert('L')
# i.show()
# b = np.array(i)
# print(b)
# c = img.fromarray(b)
# c.show()
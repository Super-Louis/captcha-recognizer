# -*- coding: utf-8 -*-
# Author  : Super~Super
# FileName: captcha_generator.py
# Python  : python3.6
# Time    : 18-7-5 19:05
# Github  : https://github.com/Super-Louis

import random
import string
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# 字体的位置，不同版本的系统会有不同
font_path = '/home/liuchao/Documents/jianxun-resume/resume/static/fonts/arial.ttf'
# 生成几位数的验证码
number = 4
# 生成验证码图片的高度和宽度
size = (100, 30)
# 背景颜色，默认为白色
bgcolor = (255, 255, 255)
# 字体颜色，默认为蓝色
fontcolor = (0, 0, 255)
# 干扰线颜色。默认为红色
linecolor = (255, 0, 0)
# 是否要加入干扰线
draw_line = True
# 加入干扰线条数的上下限
line_number = (1, 5)

# 用来随机生成一个字符串
source = list(string.ascii_letters)
for index in range(0, 10):
    source.append(str(index))
letter_exclude = ['I','i','j', 'l','1', '0', 'O', 'o', 'w', 'c', 'k', 's', 'v', 'x', 'z']
final_source = []
for i in source:
    if i not in letter_exclude:
        final_source.append(i)


def gene_text():
    return ''.join(random.sample(final_source, number))  # number是生成验证码的位数


# 用来绘制干扰线
def gene_line(draw, width, height):
    begin = (random.randint(0, width), random.randint(0, height))
    end = (random.randint(0, width), random.randint(0, height))
    draw.line([begin, end], fill=linecolor)


# 生成验证码
def gene_code(num, once=False):
    width, height = size  # 宽和高
    image = Image.new('RGBA', (width, height), bgcolor)  # 创建图片
    font = ImageFont.truetype(font_path, 25)  # 验证码的字体
    draw = ImageDraw.Draw(image)  # 创建画笔
    text = gene_text()  # 生成字符串
    text='9hxm'
    font_width, font_height = font.getsize(text)
    print(font_width, font_height)
    draw.text(((width - font_width) / number, (height - font_height) / number), text,
              font=font, fill=fontcolor)  # 填充字符串
    if draw_line:
        gene_line(draw, width, height)
    # image = image.transform((width+30,height+10), Image.AFFINE, (1,-0.3,0,-0.1,1,0),Image.BILINEAR)  #创建扭曲
    i = random.choice([-1, 1])
    # image = image.transform((width + 20, height + 10), Image.AFFINE,
    #                         (1, i * 0.3, 0, i * 0.1, 1, 0), Image.BILINEAR)  # 创建扭曲
    # image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)  # 滤镜，边界加强
    # image = image.filter(ImageFilter.BLUR) # 模糊处理
    image = image.rotate(random.uniform(-10, 10))
    # 将倾斜后的图片paste至空白图框
    image_out = Image.new('RGB', (120, 40), (255, 255, 255))
    image_out.paste(image, (15, 5))

    # 将paste后多余的像素点像素置为255
    draw_out = ImageDraw.Draw(image_out)
    for x in range(120):
        for y in range(40):
            c = image_out.getpixel((x, y))
            if c == (0, 0, 0):
                draw_out.point((x, y), fill=(255, 255, 255))
    # image_out = image_out.filter(ImageFilter.EDGE_ENHANCE_MORE)
    # 图片模糊处理
    image_out = image_out.filter(ImageFilter.GaussianBlur(1))
    save_path = '{}_{}.png'.format(text, num) if once else '../data/generated_images/{}_{}.png'.format(text, num)
    image_out.save(save_path)  # 保存验证码图片
    print('count: {}, image: {}'.format(num, text))


if __name__ == "__main__":
    # for i in range(100000):
    #     gene_code(i)
    # im = Image.open('YgN4.png')
    # im.show()
    gene_code(1, once=True)

# 导入smtplib模块
import smtplib
import os

# 从email.mime.multipart中导入MIMEMultipart类
from email.mime.multipart import MIMEMultipart
# 从email.header中导入Header类
from email.header import Header

# 从email.mime.text中导入MIMEText类
from email.mime.text import MIMEText

# 从email.mime.image中导入MIMEImage类
from email.mime.image import MIMEImage

# 1、连接邮箱服务器
# 连接邮箱服务器：连接邮箱服务器：使用smtplib模块的类SMTP_SSL，创建一个实例对象qqMail
qqMail = smtplib.SMTP_SSL("smtp.qq.com", 465)

# 2、登陆邮箱
# 设置登录邮箱的帐号为："zhangxiaofan@qq.com"，赋值给mailUser
mailUser = "553159409@qq.com"
# 将邮箱授权码"xxxxx"，赋值给mailPass
mailPass = "yoxwatwwjliubedg"
# 登录邮箱：调用对象qqMail的login()方法，传入邮箱账号和授权码
qqMail.login(mailUser, mailPass)

# 3、编辑收发件人
# 设置发件人和收件人
sender = "553159409@qq.com"
receiver = "553159409@qq.com"
# 使用类MIMEMultipart，创建一个实例对象message
message = MIMEMultipart()
# 将主题写入 message["Subject"]
message["Subject"] = Header("SLURM server got!")
# 将发件人信息写入 message["From"]
message["From"] = Header(f"slurm<{sender}>")
# 将收件人信息写入 message["To"]
message["To"] = Header(f"yuange<{receiver}>")

# 4、构建正文
# 设置邮件的内容，赋值给变量textContent
textContent = "See you then!"
# 编辑邮件正文：使用类MIMEText，创建一个实例对象mailContent
mailContent = MIMEText(textContent, "plain", "utf-8")

# # 将文件路径，赋值给filePath
# filePath = r"D:\学习资料\Python\test\zhangandlu.jpg"
# # 使用with open() as语句以rb的方式，打开路径为filePath的图片，并赋值给imageFile
# with open(filePath, "rb") as imageFile:
#     # 使用read()函数读取文件内容，赋值给fileContent
#     fileContent = imageFile.read()

# # 5、设置附件
# # 设置邮件附件：使用类MIMEImage，创建一个实例对象attachment
# attachment = MIMEImage(fileContent)
# # 调用add_header()方法，设置附件标题
# attachment.add_header("Content-Disposition", "attachment", filename="合照.jpg")

# 添加正文：调用对象message的attach()方法，传入正文对象mailContent作为参数
message.attach(mailContent)
# # 添加附件：调用对象message的attach()方法，传入附件对象attachment作为参数
# message.attach(attachment)

# 6、发送邮件
# 发送邮件：使用对象qqMail的sendmail方法发送邮件
if os.environ.get('SLURM_PROCID') is not None:
    if int(os.environ.get('SLURM_PROCID')) == 0:
        qqMail.sendmail(sender, receiver, message.as_string())
        print("E-mail sent successfully!")
    else:
        pass # do nothing
else:
    qqMail.sendmail(sender, receiver, message.as_string())
    print("E-mail sent successfully!")
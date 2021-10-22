# 创建本地仓库并且进行远程连接github仓库

* 在本地创建一个文件夹（folder）用来存储文件

**重要提示，github在2021年8月13号之后就无法使用用户名和密码来进行远程push文件，只能使用ssh来进行远程连接**

* `ssh-keygen -t rsa` 生成rea key
* 然后一直按回车，直到出现rea key，并进行复制
* `cat /Users/**jiayue/.ssh/id_rsa.pub` 创建本地ssh连接，并且复制该连接
* 在github上点击头像->setting->SSH and GPG keys->new SSH keys->title随便取->将ssh本地连接粘贴到key中，并保存
* 进入本地仓库路径，`ssh -T git@github.com` 进行远程连接
* 使用`git add 文件名` 将文件暂存缓存区
* 使用`git commit -m "title名" 将文件提交到本地仓库`
* `git remote add origin git@github.com:NBd-hub/Deep-learning.git` 使用ssh进行远程连接
* `git push -u origin master` 进行提交到远程仓库
* **tips** ：`git remote rm origin ` 取消关联仓库     `git status 查看状态` `git ls-files` 查看本地已提交的仓库文件


# 代码贡献指南

1. 首先非常欢迎和感谢对本项目发起Pull Request的同学。

1. 本项目代码风格为使用4个空格代表一个Tab，因此在提交代码时请注意一下，否则很容易在IDE格式化代码后与原代码产生大量diff，这样会给其他人阅读代码带来极大的困扰。本人使用的是vsCode编辑器，可供参考。

1. **提交代码前，请检查代码是否已经格式化，并且保证新增加或者修改的方法都有完整的参数说明，而public方法必须拥有相应的单元测试并通过测试。**

1. 本项目可以采用的接受代码贡献的方式：

- 基于[Git Flow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow)开发流程，因此在发起Pull Request的时候请选择develop分支，详细步骤参考后文，推荐使用此种方式贡献代码。


### PR方式贡献代码步骤

* 在 GitHub 上 `fork` 到自己的仓库，如 `vice_jin/face_recognition_websites`，然后 `clone` 到本地，并设置用户信息。

```bash

$ git clone git@github.com:vice-jin/face_recogniton_websites.git

$ cd face_recogniton_websites

$ git config user.name "yourname"

$ git config user.email "your email"

```

* 修改代码后提交，并推送到自己的仓库。

```bash

$ #do some change on the content

$ git commit -am "Fix issue #1: change something"

$ git push

```

* 在 GitHub 网站上提交 Pull Request。

* 定期使用项目仓库内容更新自己仓库内容。

```bash

$ git remote add upstream git@github.com:vice-jin/face_recogniton_websites.git

$ git fetch upstream

$ git checkout develop

$ git rebase upstream/develop

$ git push -f origin develop

```


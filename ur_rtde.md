# smart_robot_platform_cmake

#### 介绍
基于视觉理解机器人智能控制系统，适用于UR系列机器人

#### 软件架构
软件采用QT应用程序开发框架编写，使用CMAKE工具构建项目

#### 开发平台

- ubuntu 16.04
- QT 5.14.2


#### 安装教程

##### 1.  从gitee代码仓库clone项目
`git clone git@gitee.com:mr-tong/smart_robot_platform_cmake.git`
##### 2.  安装[ur_rtde](https://gitlab.com/sdurobotics/ur_rtde)机器人实时控制工具包

```
sudo add-apt-repository ppa:sdurobotics/ur-rtde
sudo apt-get update
sudo apt install librtde librtde-dev
```
 **注：如以上apt-get update命令出现404无法访问 错误，用以下方式手动安装**
- 在[软件包源地址](http://ppa.launchpad.net/sdurobotics/ur-rtde/ubuntu/pool/main/u/ur-rtde/)下载librtde librtde-dev软件包.deb文件
- 往您的 `/etc/apt/sources.list` 文件中像下面这样添加一行：`deb http://ftp.de.debian.org/debian sid main `
- 执行以下命令，删掉旧版本依赖包、更新新版本依赖包、安装boost-thread包、手动安装librtde librtde-dev

```
sudo apt-get remove libstdc++6 librtde librtde-dev
sudo apt-get install --only-upgrade libstdc++6 libc6 libc6-dbg
sudo apt-get install libboost-thread1.71.0
sudo dpkg -i librtde-dev_1.3.2-1_amd64.deb  librtde_1.3.2-1_amd64.deb
```
##### 3.  安装boost库
- 这里是列表文本下载[boost](https://www.boost.org/)
- 解压.bz2
`tar -jxvf xx.tar.bz2`
- 安装
```
./bootstrap.sh
sudo ./b2
sudo ./b2 install
```
#### 使用说明

1.  进入工程
`cd smart_robot_platform_cmake`
2.  cmake构建
`cmake ./`
3.  编译
`make`
4.  运行
`./smart_robot_platform`

#### 参与贡献 

1.  Fork 本仓库
2.  新建 Feat_xxx 分支
3.  提交代码
4.  新建 Pull Request
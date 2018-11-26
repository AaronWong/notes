# Install Shadowsocks in Ubuntu

## 方法一：

```shell
$ sudo apt-get update
$ sudo apt-get install python-pip
$ sudo apt-get install python-setuptools m2crypto
$ sudo apt-get install shadowsocks
```

安装后需要对Shadowsocks进行配置，在/etc目录下新建shadowsocks.json文件，添加以下内容

```shell
vi /etc/shadowsocks.json
```

```json
{
	"server": "0.0.0.0",
	"server_port": 443,
	"local_address": "127.0.0.1",
	"local_port": 1080,
	"password": "your password",
	"method": "aes-256-cfb",
	"fast_open": true,
	"workers": 1
}
```

启动shadowsocks
```shell
$ ssserver -c /etc/shadowsocks.json
$ nohup ssserver -c /etc/shadowsocks.json &
```

为了不每次启动都需手动输入一遍，设置为开机启动。在/etc/rc.local中添加如下命令，注意在exit 0之前。
```shell
vi /etc/rc.local
```
`nohup ssserver -c /etc/shadowsocks.json &`



## 方法二：

安装shadowsocks-qt5 GUI
```shell
$ sudo add-apt-repository ppa:hzwhuang/ss-qt5
$ sudo apt-get update
$ sudo apt-get install shadowsocks-qt5
```
Firefox浏览器 下载FoxyProxy插件 
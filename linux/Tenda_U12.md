# Tenda U12 驱动安装

## 1 下载

下载地址： https://github.com/gnab/rtl8812au

```shell
git clone https://github.com/gnab/rtl8812au.git
```

## 2 方法一：

### Building

```shell
$ cd 目录
$ sudo make
$ sudo insmod 8812au.ko
```

### Installing

```shell
$ sudo cp 8812au.ko /lib/modules/$(uname -r)/kernel/drivers/net/wireless
$ sudo depmod
```
当 ubuntu 内核升级时，需要再次make, cp, and depmod



## 3 方法二：

### install dkms

```shell
$ sudo apt-get install build-essential dkms 
```

### Then add it to DKMS:

```shell
$ sudo dkms add -m 8812au -v 4.2.2
$ sudo dkms build -m 8812au -v 4.2.2
$ sudo dkms install -m 8812au -v 4.2.2
```

### Check with:

```shell
$ sudo dkms status
```

### Automatically load at boot:

```shell
$ echo 8812au | sudo tee -a /etc/modules
```

### Eventually remove from DKMS with:

```shell
$ sudo dkms remove -m 8812au -v 4.2.2 --all
```


---
layout: post
title: STLink Installation on Ubuntu 14.04
category: Software Installation
---

**STLink Installation on Ubuntu 14.04**

In Linux a command line program can be used to program STM32 microcontrollers via the embedded ST-LINK found on evaluation boards such as the STM32F4 discovery .

In this tutorial, a STM32F4 discovery board is programmed in Ubuntu Linux 14.04

**install dependencies**
 First install dependencies `git` and `libusb`

```
sudo apt-get install libusb-1.0-0-dev git
```

**Build stlink binaries**

In the terminal window, change to a suitable directory to work from and enter:

> git clone https://github.com/texane/stlink stlink.git

This will clone the `stlink` github repository to the working directory

Change to the new directory and then make the project. Enter these two lines in the terminal window:

```
$cd stlink.git
$ ./autogen.sh
$ ./configure
$ make

```

**Copy the st-flash file to the file system.**

In the `stlink.git` directory contains the  contains the `st-flash` program after compilation
Copy this to systems binary path

```
sudo cp st-flash /usr/bin
```

**UDEV Permissions**
Set up udev rules so that it is possible to run st-flash without using the `sudo` command. Change back to the `stlink.git` directory and then copy the rules files to the file system. In the terminal window enter:

```
cd ..
sudo cp *.rules /etc/udev/rules.d
sudo restart udev
```


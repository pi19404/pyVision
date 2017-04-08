---
layout: post
title: Install Docker on Ubuntu 14.04
category: Software Installation
---


**Install Docker**

Docker requires a 64-bit installation regardless of your Ubuntu version. Additionally, your kernel must be 3.10 at minimum. The latest 3.10 minor version or a newer maintained version are also acceptable.

To check your current kernel version, open a terminal and use uname -r to display your kernel version:

```
uname -r
```

**Update your apt sources**

Docker’s APT repository contains Docker 1.7.1 and higher. To set APT to use packages from the new repository:

```
sudo apt-get install apt-transport-https ca-certificates
```

**Add the new GPG key.**

```
$ sudo apt-key adv --keyserver hkp://p80.pool.sks-keyservers.net:80 --recv-keys 58118E89F3A912897C070ADBF76221572C52609D
```

**Apt sources**

Add the line 

```
deb https://apt.dockerproject.org/repo ubuntu-trusty main
```

to  `/etc/apt/sources.list.d/docker.list` file

**Prerequisites by Ubuntu Version**

Install the following pre-requisites

```
sudo apt-get install linux-image-extra-$(uname -r)
sudo apt-get install apparmor
```

**Install Docker**

Run the below command to install docker

```
sudo apt-get install docker-engine
```

Start the docker daemon

```
sudo service docker start
```


verify the docker is installed properly

```
sudo docker run hello-world
```


**Uninstallation**

```
 sudo apt-get purge docker-engine
 sudo apt-get autoremove --purge docker-engine
 rm -rf /var/lib/docker
```

## **References**

- https://docs.docker.com/engine/installation/linux/ubuntulinux/



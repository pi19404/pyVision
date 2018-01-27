---
layout: post
title: Adding Swap Space on Ubuntu 14.04
published: true
---

Swap space in Linux can be used when a system requires more memory than it has been physically allocated. When swap space is enabled, Linux systems can swap infrequently used memory pages from physical memory to swap space (either a dedicated partition or a swap file in an existing file system) and free up that space for memory pages that require high speed access.

To see if your instance is using swap space, you can use the `swapon -s` command.

If you don't see a swap volume listed with this command, you may need to enable swap space for the device. Check your available disks using the `lsblk` command.

```
root@ip-172-31-55-140:/home/ubuntu# lsblk
NAME    MAJ:MIN RM SIZE RO TYPE MOUNTPOINT
xvda    202:0    0   8G  0 disk 
└─xvda1 202:1    0   8G  0 part /
xvdf    202:80   0   4G  0 disk 
root@ip-172-31-55-140:/home/ubuntu# 
```

Here, the swap volume xvdf is available to the instance, but it is not enabled (notice that the MOUNTPOINT field is empty). You can enable the swap volume with the swapon command.

```
mkswap /dev/xvdf
sudo swapon /dev/xvda3
```

You will also need to edit your /etc/fstab file so that this swap space is automatically enabled at every system boot.

```
/dev/xvdf      none    swap    sw  0       0
```


**References**

- http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instance-store-swap-volumes.html

---
layout: post
title: Accessing VMWareWorkstation machine via VNC
category: Software Installation
---

You can use a VNC client to connect to a running virtual machine. Because VNC software is cross-platform, you can use virtual machines running on different types of computers.

Workstation does not need to be running to use VNC to connect to a virtual machine. Only the virtual machine needs to be running, and it can be running in the background.
When you use a VNC client to connect to a virtual machine, some features do not work or are not available.

- You cannot take or revert to snapshots.
- You cannot power on, power off, suspend, or resume the virtual machine. You can shut down the guest operating system. Shutting down might power off the virtual machine.
- You cannot copy and paste text between the host system and the guest operating system.
- You cannot change virtual machine settings.
- Remote display does not work well if you are also using the 3D feature.


## **Configuring a Virtual Machine as a VNC Server** ##

You can use Workstation to configure a virtual machine to act as a VNC server so that users on other computers can use a VNC client to connect to the virtual machine. You do not need to install specialized VNC software in a virtual machine to set it up as a VNC server.

	
1. 	 Select the virtual machine and select VM > Settings.	
2.   On the Options tab, select VNC Connections and select Enable VNC.	
3.   To allow VNC clients to connect to multiple virtual machines on the same host system, specify a unique port number for each virtual machine.
4.   Use should use a port number in the range from 5901 to 6001. Other applications use certain port numbers, and some port numbers are privileged.
5.   (Optional) Set a password for connecting to the virtual machine from a VNC client.
6.   (Optional) Click View VNC Connections to see a list of the VNC clients that are remotely connected to the virtual machine and find out how long they have been connected.
7.   Click OK to save your changes.

	
## **Install a VNC client on your computer.** ##

Download the RealVNC Viewer from following link
[https://www.realvnc.com/download/viewer/](https://www.realvnc.com/download/viewer/)



# **Start the VNC client on your computer.** #

The IP address is of the host machine running the VMWare guest operating system

In the present case it is `10.0.0.17` and port number configured in VMWare workstation settings is 5910

	
It might take multiple attemtps to connect if the internet speed is slow or if the connection is being done over a WifiNetwork


## **References** ##

- https://pubs.vmware.com/workstation-9/topic/com.vmware.ws.using.doc/GUID-FB23927B-98A0-45E9-BFAC-85152F14BCAC.html
- https://pubs.vmware.com/workstation-9/index.jsp?topic=%2Fcom.vmware.ws.using.doc%2FGUID-FB23927B-98A0-45E9-BFAC-85152F14BCAC.html
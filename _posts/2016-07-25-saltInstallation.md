---
layout: post
title: Install and Configure Salt Master and Minion Servers on Ubuntu 14.04
category: Software Installation
---

## **Introduction**

`SaltStack` is a powerful, flexible, high performing configuration management and remote execution system. It can be used manage your infrastructure from a centralized location while minimizing manual maintenance steps.


If you are setting up your environment for the first time, you should install a Salt master on a dedicated management server or VM, and then install a Salt minion on each system that you want to manage using Salt.

## **Installation**

**Install the Master Daemon**

Packages for Ubuntu are also published in the saltstack PPA. If you have the add-apt-repository utility, you can add the repository and import the key in one step:

```
sudo apt-get install python-software-properties
sudo apt-get install software-properties-common

echo deb http://repo.saltstack.com/apt/ubuntu/14.04/amd64/latest trusty main | sudo tee /etc/apt/sources.list.d/saltstack.list 
wget -O - https://repo.saltstack.com/apt/ubuntu/14.04/amd64/latest/SALTSTACK-GPG-KEY.pub | sudo apt-key add -
sudo apt-get update
```
Install the salt-minion, salt-master, or other Salt components:

```
sudo apt-get install salt-master
sudo apt-get install salt-minion
sudo apt-get install salt-ssh
sudo apt-get install salt-syndic
sudo apt-get install salt-cloud
sudo apt-get install salt-api
```

## **Docker Salt master Configuration**

We will use a docker image for salt-master.

Download the salt-master docker image from https://hub.docker.com/r/bbinet/salt-master/

Run the below command to start the salt-master docker image

```
docker run -d --name klugadmin-salt-master \
    -v /home/klugadmin/pi/docker/salt-master/config:/config \
    -v /home/klugadmin/pi/docker/salt-master/data:/data \
    -p 4505:4505 \
    -p 4506:4506 \
    -p 443:443 \
    bbinet/salt-master
```
 
To login to salt master shell in docker container run the command

```
sudo docker exec -i -t klugadmin-salt-server /bin/bash
```

**Get the Salt Master Public Key Fingerprint**

Before we begin, we should grab the Salt master's key fingerprint. We can add this to our minion configuration for increased security.

On your Salt master server, type:

```
sudo salt-key -F master
```

The output should look something like this:

```
root@ecf1f39f73a3:/# salt-key -F master
Local Keys:
master.pem:  f3:67:2a:06:a2:70:21:f9:3f:c0:8c:0f:cc:e6:5e:41
master.pub:  7b:33:b2:4e:a0:f4:cd:b8:b4:4f:de:02:a7:26:d5:bc
Accepted Keys:
klugserver:  c8:de:11:56:a7:03:76:58:fd:43:ed:3e:61:d4:48:7a
root@ecf1f39f73a3:/#
```
The value of the `master.pub key`, located under the "Local Keys" section is the fingerprint we are looking for. Copy this value to use in our Minion configuration.

## **Salt-Minion Installation and configuration**

We will install the salt-minion on each of the systems that needs to be managed

**Modify the Minion Configuration**

Back on your new Salt minion, open the minion configuration file with sudo privileges:

```
sudo vi/etc/salt/minion
```

We need to specify the location where the Salt master can be found. This can either be a resolvable DNS domain name or an IP address:

```
/etc/salt/minion
master: ip_of_salt_master
```

Next, set the master_finger option to the fingerprint value you copied from the Salt master a moment ago:

```
/etc/salt/minion
master_finger: '7b:33:b2:4e:a0:f4:cd:b8:b4:4f:de:02:a7:26:d5:bc'
```
Save and close the file when you are finished.

Now, restart the Salt minion daemon to implement your new configuration changes:
```
sudo restart salt-minion
```
The new minion should contact the Salt master service at the provided address. 

It will then send its key for the master to accept. In order to securely verify the key, need to check the key fingerprint on the new minion server.

To do this, type:

```
sudo salt-call key.finger --local
```
You should see output that looks like this:

```
klugadmin@klugserver:~$ sudo salt-call key.finger --local
local:
    c8:de:11:56:a7:03:76:58:fd:43:ed:3e:61:d4:48:7a
```

You will need to verify that the key fingerprint that the master server received matches this value.


**Accept the Minion Key on the Salt Master**

Back on your Salt master server, we need to accept the key.

First, verify that we have an unaccepted key waiting on the master:

```
sudo salt-key --list all
```

You should see a new key in the "Unaccepted Keys" section that is associated with your new minion:

```
Output
Accepted Keys:
klugserver
Denied Keys:
Unaccepted Keys:
saltminion
Rejected Keys:
```

Check the fingerprint of the new key. Modify the highlighted portion below with the minion ID that you see in the `Unaccepted Keys` section:

```
sudo salt-key -f klugserver
```

The output should look something like this:

```
Output
Unaccepted Keys:
klugserver:  c8:de:11:56:a7:03:76:58:fd:43:ed:3e:61:d4:48:7a
```

If this matches the value you received from the minion when issuing the salt-call command, you can safely accept the key by typing:

```
sudo salt-key -a klugserver
```

The key should now be added to the `Accepted Keys` section:

```
sudo salt-key --list all
Output
Accepted Keys:
klugserver
Denied Keys:
Unaccepted Keys:
Rejected Keys:
```


Test that you can send commands to your new minion by typing:
```
root@ecf1f39f73a3:/# salt '*' test.ping
klugserver:
    True
root@ecf1f39f73a3:/#
```

You should receive back answers from the minion daemons you've configured:

```
root@ecf1f39f73a3:/# salt '*' test.ping
klugserver:
    True
```

You should now have a Salt master server configured to control your infrastructure. We've also walked through the process of setting up a new minion server. You can follow this same procedure for additional Salt minions. These are the basic skills you need to set up new infrastructure for Salt management.


**References**

- https://docs.saltstack.com/en/latest/topics/installation/index.html
- https://www.digitalocean.com/community/tutorials/how-to-install-and-configure-salt-master-and-minion-servers-on-ubuntu-14-04
- https://www.openstack.org/summit/tokyo-2015/videos/presentation/chef-vs-puppet-vs-ansible-vs-salt-whats-best-for-deploying-and-managing-openstack
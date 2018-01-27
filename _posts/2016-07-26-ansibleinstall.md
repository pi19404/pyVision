---
layout: post
title: Ansible Installation on Ubuntu 14.04 and Installation of Netbeans IDE
category: Software Installation
---


## **Ansible installation**

In this guide, we will discuss how to install Ansible on an Ubuntu 14.04 machine and go over some basics of how to use the software.

Ansible by default manages machines over the SSH protocol.

Once Ansible is installed, it will not add a database, and there will be no daemons to start or keep running. You only need to install it on one machine (which could easily be a laptop) and it can manage an entire fleet of remote machines from that central point. When Ansible manages remote machines, it does not leave software installed or running on them, so there’s no real question about how to upgrade Ansible when moving to a new version.

Currently Ansible can be run from any machine with Python 2.6 or 2.7 installed (Windows isn’t supported for the control machine).


Ansible also uses the following Python modules that need to be installed


To configure the PPA on your machine and install ansible run these commands:

```
$sudo apt-get update
$sudo apt-get install openssh-server
$sudo apt-get install software-properties-common
$sudo apt-add-repository ppa:ansible/ansible
$ sudo apt-get update
$ sudo apt-get install ansible

```

## **Adding Ansible clients**

Now that you’ve installed Ansible, it’s time to get started with some basics.

Edit (or create) `/etc/ansible/hosts` and put one or more remote systems in it. Your public SSH key should be located in `authorized_keys` on those systems:

```
10.2.1.98
aserver.example.org
bserver.example.org
```

To set up SSH agent to avoid retyping passwords, you can do:

```
ssh-agent bash
ssh-add ~/.ssh/id_rsa
```

We need to add the servers ssh public key on the clients authorized hosts

To generate the ssh public/private key pair

```
$ssh-keygen
```

Follow instructions to generate the keys

Copy the servers public key  and enter in on the hosts
`authorized_keys` file on clients machine


```
#on the host
cat ~/.ssh/id_rsa.pub
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQCUUKR4c6AEubtkiVToENFyxMiOUQ+P2yjcJlo3167l5MFX7d4wfW0nhI1lteSdBVmVJL0ZRfNzD5EpzSCSLummXw5m0O4jNtEVweVNA1h57ogqELZ8p9JE0hsA2lLAPIFUmC3uBG1oK18o8rtaxNCdC/h575db9CeGB/lkL8PsnRLZzX/522BdNNzpgOQGPSArvl/ChHbZ2NgBHxXlRkxUQeFYbwPCamNa4BztDhvvJbjpyQQj5ULj7oiwE6BmNSVrsSO3QoI2I7RHKpfQNZwPTMlRE1V1h9r7UUcd/E96TMoRZXhDu8IzvWY8zNoyC4ErmCtKOkO8ocd/Xp+CkUEP root@gui-ubuntu

```


Add/append the text to the `authorized_keys` file found in `~root/.ssh ` directory

Now you should be able to log into the client systems via ssh without requiring any passwords


## **Running First Ansible command**

This command should ping all the hosts configured in the  `/etc/ansible/hosts`

```
$ ansible all -m ping
```

we can also execute commands on the client machines by specifically mentioning client machine hostname/ipaddress (`localhost`)

```
$ ansible localhost -a "/bin/echo hello"
```

## **Ansible Playbooks**


Since we can execute any command on the remote system,any predefined installation/configuration steps can be executed across several machines in an ordered and repeatable manner.

Playbooks are Ansible’s configuration, deployment, and orchestration language. They can describe a policy you want your remote systems to enforce, or a set of steps in a general IT process.

At a basic level, playbooks can be used to manage configurations of and deployments to remote machines.Playbooks are designed to be human-readable and are developed in a basic text language. 

Playbooks are expressed in YAML format (see YAML Syntax) and have a minimum of syntax, which intentionally tries to not be a programming language or script, but rather a model of a configuration or a process.

We will look at using existing playbooks

## **Ansible Galaxy** 

Ansible galaxy hosts `ansible roles` for configurations for most softwares

we will look at installing oracle JDK using ansible playbook

https://galaxy.ansible.com/tersmitten/oracle-java/ provides details on `ansible roles` for oracle JDK

To fetch the `ansible roles ` on the host ansible machine 

```
ansible-galaxy install tersmitten.oracle-java
```

Example playbook file 

we write this sample playbook which which instructs ansible to install the role `tersmitten.oracle-java` on the `localhost` machine

```
- hosts: servers
  roles:
     - { role: tersmitten.oracle-java }
```

to run the playbook we run

```
ansible-playbook orable_java.playbook
```

This will by default install `oracle jdk7` on the client machine

Similarly by installing the roles and configuring playbook any software can be installed on client machine

To install netbeans IDE

```
ansible-galaxy install jeqo.netbeans
```

Playbook file for the same

```
- hosts: localhost 
  roles:
     - { role: jeqo.netbeans}
  vars :
    netbeans_version : 8.1 
    netbeans_base: "\{\{ /opt/softwares/netbeans }}"
    netbeans_home: "{{ netbeans_base }}/netbeans-{{ netbeans_version }}"


```

We can also pass variables via playbook if the roles are written to support them


## **References**

- http://docs.ansible.com/ansible/intro_getting_started.html
- http://docs.ansible.com/ansible/intro_getting_started.html
- http://docs.ansible.com/ansible/playbooks_intro.html




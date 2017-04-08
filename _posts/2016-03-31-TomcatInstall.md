---
layout: post
title: Install Apache Tomcat 7 on Ubuntu 14.04
category: Software Installation
---

In this we will look at installing
Apache Tomcat 7 on Ubuntu 14.04 and deplyoying a web application


# Install Tomcat with apt-get #

This will install the tomcat software and administration tools

```
sudo apt-get install tomcat7
sudo apt-get install tomcat7-admin

sudo chgrp -R tomcat7 /etc/tomcat7
sudo chmod -R g+w /etc/tomcat7 

```

#Directory Strcuture

- **CATALINA_HOME**

The tomcat installation directory is referred as
CATALINA_HOME and is set to `/usr/share/tomcat7`
by default


- **CATALINA_BASE**

This tomcat runtime directory and is set to `/var/lib/tomcat7` by default
web applications can be found here. 

webapps can be found in `webapps` directory

- **CONFIGURATION**

Configuration files can found at `/etc/tomcat7`


- **DEFAULT CONFIGURATION**

default configuration files for tomcat can be found at
`/etc/default/tomcat7`


## **COMMANDS** ##


Tomcat service can be checked 

```
sudo service tomcat7 status
```

Commands for starting and stopping are as follows


```
sudo service tomcat7 start
sudo service tomcat7 stop
```

## **CONFIGURATION** ##

- PORT CONFIGURATION

By default Tomcat runs a HTTP connector on port 8080
This ports can be configured by changing the following lines in `/etc/tomcat7/server.xml`

```
<Connector port="8090" protocol="HTTP/1.1" 
               connectionTimeout="20000" 
               redirectPort="8443" />
````               

- JVM CONFIGURATION


By default Tomcat will run preferably with OpenJDK JVMs, then try the Sun JVMs, then try some other JVMs. You can force Tomcat to use a specific JVM by setting JAVA_HOME in `/etc/default/tomcat7`:

```
JAVA_HOME=/usr/lib/jvm/java-6-sun
```


- Declaring users and roles

Usernames, passwords and roles (groups) can be defined centrally in a Servlet container. This is done in the `/etc/tomcat7/tomcat-users.xml` file:

```
<role rolename="admin"/>
<user username="tomcat" password="reloded23" roles="admin,manager-gui,host-gui"/>
```

- Larger uploads

Modify `/usr/share/tomcat7-admin/manager/WEB-INF/web.xml` to handle larger uploads.


```
    <multipart-config>
      <max-file-size>72428800</max-file-size>
      <max-request-size>72428800</max-request-size>
      <file-size-threshold>0</file-size-threshold>
    </multipart-config>
    
```

## **Deploying the Web Application** ##

Tomcat manager GUI can be accessed as follows

```

http://10.0.0.15:8090/manager/html

```

you can manager all the webapps from this terminal



## **References** ##

- http://askubuntu.com/questions/135824/what-is-the-tomcat-installation-directory
- https://help.ubuntu.com/lts/serverguide/tomcat.html
- http://mythinkpond.wordpress.com/2011/07/01/tomcat-6-infamous-severe-error-listenerstart-message-how-to-debug-this-error/
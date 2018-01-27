---
layout: post
title: Tomcat7 installation on Ubuntu 14.04
published: true
---

**tomcat7 installation**

```
sudo apt-get install tomcat7
sudo apt-get install tomcat7-docs tomcat7-admin tomcat7-examples
```

**To start the service run **

```
service tomcat7 start
service tomcat7 stop
service tomcat7 restart
```

tomcat7 is available by default on port 8080


Tomcat is installed with CATALINA_HOME in /usr/share/tomcat7 and CATALINA_BASE in /var/lib/tomcat7, following the rules from /usr/share/doc/tomcat7-common/RUNNING.txt.gz.


**tomcat-users.xml**

Tomcat configuration files are found in the directory: CATALINA_HOME/conf (where CATALINA_HOME environment variable is the Tomcat installation directory). The main configuration file is server.xml. tomcat-users.xml is one of the configuration files

manager-gui roles needs to be added and assigned to a user to get access to tomcat admin panel

```
<tomcat-users>
    <role rolename="manager-gui"/>
    <role rolename="admin-gui"/>
    <role rolename="tomcat"/>
    <user password="password" roles="manager-gui,admin-gui,admin" username="admin"/>
    <user password="tomcat" roles="manager-gui,admin-gui,manager-script,admin" username="tomcat"/>
    <user password="tomcat7" roles="manager-script,admin" username="tomcat7"/>
</tomcat-users>

```

**Create a softlink to conf directory **

Some softwares exepct the conf directory to be present in the CATALINA_HOME but is present in CATALINA_BASE .In which case we need to create a softlink for the same

```
ln -s /var/lib/tomcat7/conf /usr/share/tomcat7/conf
ln -s /var/lib/tomcat7/logs /usr/share/tomcat7/logs
```




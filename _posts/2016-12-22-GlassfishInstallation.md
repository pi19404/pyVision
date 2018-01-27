---
layout: post
title: Glassfish Installation on Ubuntu 14.04
published: true
---

**Download glassfish**

```
wget http://download.java.net/glassfish/4.1/release/glassfish-4.1.zip
unzip glassfish-4.1.zip
rm -f glassfish-4.1.zip
```

**Create the following startup script**

```
#
# description: Startup script for Glassfish Application Server
# processname: glassfish
 
GLASSFISH_HOME=/opt/glassfish4/glassfish
export GLASSFISH_HOME
GLASSFISH_USER=glassfish
export GLASSFISH_USER
 
start() {
echo -n "Starting Glassfish: "
su $GLASSFISH_USER -c "$GLASSFISH_HOME/bin/asadmin start-domain domain1"
sleep 2
echo "done"
}
 
stop() {
echo -n "Stopping Glassfish: "
su $GLASSFISH_USER -c "$GLASSFISH_HOME/bin/asadmin stop-domain domain1"
echo "done"
}
 
case "$1" in
start)
start
;;
stop)
stop
;;
restart)
stop
start
;;
*)
echo $"Usage: glassfish {start|stop|restart}"
exit
esac
```

**Make it a executable**

```
chmod 755 /etc/init.d/glassfish
```

**Configure the user**

```
useradd glassfish
chown -R glassfish:glassfish /opt/glassfish4
```

**Glassfish administration**

```
sudo /etc/init.d/glassfish start
### Stop GlassFish ###
sudo /etc/init.d/glassfish stop
### restart GlassFish ###
sudo /etc/init.d/glassfish restart
```


**Accessing Glassfish**

GlassFish will be available on HTTP port 8080 by default also port 4848 by administration. Open your favorite browser and navigate to http://yourdomain.com:8080 or http://server-ip:4848 and complete the required the steps to finish the installation. If you are using a firewall, please open port 8080 and 4848 to enable access to the control panel.

**Turn on system administration**


To turn the remote administration on and access the GlassFish admin console via web browser, execute the following commands:

```
cd /opt/glassfish4/glassfish/bin
./asadmin --user admin
asadmin> change-admin-password
./asadmin enable-secure-admin
```

Default password is blank.You can reset the password as required

** To deploy and undeploy a file from command line **

```
cd /opt/glassfish4/glassfish/bin
./asadmin deploy war-name

./asadmin undeploy war-name
```


**References**

- http://idroot.net/tutorials/how-to-install-glassfish-on-ubuntu-14-04/


---
layout: post
title: GitBucket setup on Ubuntu 14.04
category: Software Installation 

---
# GitBucket setup on Ubuntu 14.04

GitBucket is a GitHub clone powered by Scala which has easy installation and high extensibility.

we will be installing GitBucket on Windows 7

Download latest `gitbucket.war` from the [release page](https://github.com/gitbucket/gitbucket/releases).

We will deploy it on Apache Tomcat 7.0
This deployment can be done from apache admistration browser by accessing the url http://localhost:8080.

By selecting the option `Select WAR file to Upload` and then `deploy` to deploy the application on topcat
![enter image description here](http://i.imgur.com/Ivx1VQU.png)

If it is deployed successfully then we can webservice is started and we can observer `gitbucket` in the list of deployed apps.

![enter image description here](http://i.imgur.com/ImbMQ1g.png)

Tomcat will always extract the contents of a war file, to a folder of the same name
Thus context path will be choosen by default as /gitbucket

The gitbucket admin page can be accessed by url http://localhost:8080/gitbucket

![enter image description here](http://i.imgur.com/6uOnJbX.png)

The default administrator account is root and password is root

To upgrade GitBucket, only replace gitbucket.war or redeploy the war file using the TomCat Admin panel after stop GitBucket.

 All GitBucket data is stored in HOME/.gitbucket. In case of present installation it is C:/.gitbucket

To take back up GitBucket data, copy this directory to the other disk. 


## References
- https://github.com/gitbucket/gitbucket



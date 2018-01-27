---
layout: post
title: Artifactory and Maven repository Setup
category: Software installation
---


**Artifactory and Maven repository Setup**
--------------------------------

In this article we will look at maven repository management using Artifactory repository manager.


>**Introduction**

Maven is a popular build tool available for java developers.

The purpose of maven repository is to serve as an internal private repository of all software libraries used within an organization. 

The maven stores all the software libraries in a common remote store called a repository  which helps to provide a  single central reference repository of all dependent software libraries rather than several independent local libraries and reduce the duplication of dependent software libraries (jars) required to build an application .

>**Types of maven repository**

- **Local repository** – exists on developers machine and is maintained by the developer. It is in sync with the maven repositories defined in the ‘settings.xml’ in their ‘~home/.m2’ folder. 
- **Private remote internal repository** – This the repository which we will setup. We will change the maven `pom.xml` or `settings.xml` to use this repository
- **Public remote external repository** – This is the public external repository at `ibiblio.org`. By default, maven synchronizes with this repository.


>**Creating a maven repository**

This post will demonstrate how to create your own Maven2 repository and put your piece of work there. 

`JFrog Artifactory` is one of the leading open source Maven repository managers

This can be downloaded from 
https://www.jfrog.com/open-source/#os-arti

The downloaded zip files contains the web server and can be run without any other pre requisites.

However you can take the `artifactor.war` from the webapps directory and deploy it only your web server .

The application will be deployed with default context root of `/artifactory` and in our case the URL is `http://10.0.0.15:8090/artifactory/`

The default user name and password for artifactory is `admin:password`

The `artifactory manager` will provide us settings that we need to use to access the repository as target to upload/deploy user defined library as well as to access or source of dependent libraries during build. 

 Maven is configured using a `settings.xml` file located under your Maven home directory.

This will typically be 
```
/home/{user.name}/.m2/settings.xml
```

To work with Artifactory you need to configure Maven to perform the following two steps:

- Deploy artifacts to repositories through Artifactory
- Resolve artifacts through Artifactory

**Resolve artifacts through Artifactory**

To configure Maven to `resolve artifacts` through Artifactory you need to modify the `settings.xml`

To make it easy for you to configure Maven to work with Artifactory, Artifactory can automatically generate a `settings.xml` file which you can save under your Maven home directory.

The definitions in the generated `settings.xml` file override the default central and snapshot repositories of Maven.

In the `Artifact Repository Browser` of the Artifacts module, select `Set Me Up`. In the `Set Me Up` dialog, set Maven in the Tool field and click "Generate Maven Settings". You can now specify the repositories you want to configure for Maven.

Insert the contents into `settings.xml` file on the build machine

![enter image description here](http://i.imgur.com/YquSPQX.png)

The password inserted in maven in encrypted
to find the encrypted password we need to first create master password

authorized users have an additional settings-security.xml file in their ~/.m2 folder in which master password is placed

```
mvn --encrypt-master-password <password>
```

Store the output of the above command  in the ~/.m2/settings-security.xml

```
<settingsSecurity>
  <master>{jSMOWnoPFgsHVpMvz5VrIt5kRbzGpI8u+9EF1iFQyJQ=}</master>
</settingsSecurity>
```

To create user password

```
mvn --encrypt-password <password>
```

This will be used by unauthorized users to access the maven repository

Paste the password into the `password` section of `settings.xml` file.Choose the appropriate user created in Artifactory .




**Deploy 3rd party libraries**

To deploy third party libraries that you need for development environment In the `Artifacts module Tree Browser` select the repository you want to deploy  and select on `deploy` .

![enter image description here](https://i.imgur.com/ogypEg1.png)

Provide the path to the required jar files and 

![enter image description here](https://i.imgur.com/OxApSqj.png)

Once the jar file is uploaded/deployed to the repository enter the required details like `groupId`,`artifactId`,`version` etc

![enter image description here](http://i.imgur.com/ZoWSOZP.png)

Finally click on deploy to upload the `jar` file to the repository.

Once the download is complete you can browse the repository tree and navigate to the uploaded jar file.In present case the jar files are uploaded to `libs-release-local-repository`

In the dependency declaration section.You can find maven commands to be added to `pom.xml` inorder to add the dependency to the project.

![enter image description here](http://i.imgur.com/OWyIUKV.png)


Remember that you can not deploy build artifacts to remote or virtual repositories, so you should not use them in a deployment element.

**Deploy as part of build process**

To deploy build artifacts through `Artifactory` you must add a deployment element with the URL of a target local repository to which you want to deploy your artifacts.

To make this easier, `Artifactory` displays a code snippet that you can use as your deployment element. In the `Artifacts module Tree Browser` select the repository you want to deploy to and `click Set Me UP`. The code snippet is displayed under Deploy. 

![](http://i.imgur.com/hNMhcVp.png)

Copy the entire section and add to the project `pom.xml`

Once you build the project,it will be automatically deployed onto the maven repository and accessible to other users as well.

If not go to the project directory and run the commmand

```
mvn deploy
```

**References**

- https://maven.apache.org/guides/mini/guide-encryption.html
- https://devcenter.heroku.com/articles/using-a-custom-maven-settings-xml





---
layout: post
title: Importing Mysql Database on ubuntu 14.04
published: true
---

**Create a database**

```
CREATE DATABASE <DATABASENAME>;
```

**Importing a database**

```
mysql -u <username> -p<password> database_name < file.sql
```
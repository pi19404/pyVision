---
layout: post
title: Introduction to Apache Spark Lightweight Web Framework
published: true
---

Introduction
Apache Spark is a Lightweight Web Framework .Spark Framework is a true micro Java web framework. Its total size is less than a megabyte, and to keep it lean and clean we decided to cut support for Java 7 in Spark 2. If you are stuck with Java 7 for whatever reason, you unfortunately have to have to use Spark 1.

To get started create a maven project in your IDE

Add the following dependency to `pom.xml` file

```
<dependency>
    <groupId>com.sparkjava</groupId>
    <artifactId>spark-core</artifactId>
    <version>2.3</version>
</dependency>
```

**Hello World**

```
import static spark.Spark.*;

public class HelloWorld {
    public static void main(String[] args) {
        get("/hello", (req, res) -> "Hello World");
    }
}
```

This sets up a route for hello and corresponding response of "Hello Word"

```
http://localhost:4567/hello
```

In a typical RESTful application we expect to receive POST requests with json objects as part of the payload.

As far as the communication protocol goes the data is just a text.

Our job will be to check the code is well-formed JSON, that it corresponds to the expected structure, that the values are in the valid ranges, etc.

The payload or the data format should be independent of the communication protocol.
In all the apis the data will be in form of a json.This leads to losse coupling of data and commnication protocol.


**References**

- http://sparkjava.com/download.html
- https://sparktutorials.github.io/2015/04/02/hello-tutorials.html


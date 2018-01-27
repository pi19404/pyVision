---
layout: post
title: Redis Mass Insertion of Data
published: true
---


# **Redis Mass Insert**

Sometimes Redis instances need to be loaded with a big amount of preexisting or user generated data in a short amount of time, so that millions of keys will be created as fast as possible

Using a normal Redis client to perform mass insertion is not a good idea for a few reasons: the naive approach of sending one command after the other is slow because you have to pay for the round trip time for every command

Redis clients communicate with the Redis server using a protocol called RESP (REdis Serialization Protocol).
Mass insertion can be performed using redis protocol .The Redis protocol is extremely simple to generate and parse

Every redis command is represented in the following way:


**Redis command format**

```

*<args><cr><lf>
$<len><cr><lf>
<arg0><cr><lf>
<arg1><cr><lf>
...
<argN><cr><lf>
```

Where <cr> means "\r" (or ASCII character 13) and <lf> means "\n" (or ASCII character 10).

For instance the command SET key value is represented by the following protocol:

```

*3<cr><lf>
$3<cr><lf>
SET<cr><lf>
$3<cr><lf>
key<cr><lf>
$5<cr><lf>
value<cr><lf>
```

Or represented as a quoted string:

```
"*3\r\n$3\r\nSET\r\n$3\r\nkey\r\n$5\r\nvalue\r\n"
```

**Python code to generate redis protocol**

The python code to convert SET command to redis protocol string


```
def gen_redis_proto(cmd,key,value):
    proto = ""

    proto="*3"+"\r\n"

    #for arg in args1:
    arg=cmd
    proto =proto+"$"+str(len(arg))+"\r\n"
    proto =proto+arg+"\r\n"

    arg=key
    proto =proto+"$"+str(len(arg))+"\r\n"
    proto =proto+arg+"\r\n"

    arg=value
    proto =proto+"$"+str(len(arg))+"\r\n"
    proto =proto+arg+"\r\n"

    return proto
```		

For example :

```
proto=gen_redis_proto("SET","key,"value");
```

The string is written to the text file

```
"*3
$3
SET
$3
key
$5
value
```

**Loading the data to redis database**

Loading the data to redis using command line utility

```
cat test.txt | redis-cli --pipe
```



**References**

- https://redis.io/topics/mass-insert
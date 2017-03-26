---
layout: post
title: OrientDB Graph Database Installation on Ubuntu 14.04
published: true
---

**Pre-Requisites**

Java needs to be installed before processing with orientdb installation

Run the following command to install openjdk

```
sudo apt-get install openjdk-8-jre
```

**OrientDB Installation**

Download orientDB from the following page http://orientdb.com/download/

The orientdb database executables are contained within the `tar.gz` file.

Uncompress the file by running

```
gunzip orientdb-community-2.2.13.tar.gz
tar -xvf orientdb-community-2.2.13.tar
```

This will uncompress all the files in `orientdb-community-2.2.13` directory

By default orientdb is only accessible to external ip address.

To enable access to orientdb on localhost changes need to be made in `$ORIENTDB_HOME/config/orientdb-server-config.xml` file

**Starting the Server**

to start the server run the script file `$ORIENTDB_HOME/bin/server.sh`

```
2016-12-13 09:55:54:382 INFO  Listening binary connections on 0.0.0.0:2424 (protocol v.36, socket=default) [OServerNetworkListener]
2016-12-13 09:55:54:389 INFO  Listening http connections on 0.0.0.0:2480 (protocol v.10, socket=default) [OServerNetworkListener]
```

you can see that orientdb listeners are up on ports 2424 and 2480 .

Port 2480 is a web application which gives access to orientdb via browser

Port 2424 supports binary protocol which is used by API's to communicate with orientdb.

**Running Scripts**

To run scripts execute the following command

```
$ORIENDB_HOME/bin/console.sh myscript.osql
```

Example script is given below

```
connect remote:localhost:2424/emotix admin admin
create class GraphNode IF NOT EXISTS extends V ;
create property GraphNode.name IF NOT EXISTS string;
create property GraphNode.type IF NOT EXISTS string;
create property GraphNode.description IF NOT EXISTS string;

CREATE INDEX GraphNode.name UNIQUE;
```

**To export and Import Databases**

For importing the database run the following script using console

```
IMPORT DATABASE /home/ubuntu/test/emtest -preserveClusterIDs=true
```

For exporting the database run the following script using the console

```
EXPORT DATABASE /temp/petshop.export
```



** To load a CSV file to orient db database **

The orientdb ETL module is used to load data from various data sources transform it a format
suitable for orientdb and create suitable objects in orientdb database and store the information

For example if we have a csv file for a entity type person containing name and description files

we need to create a csv file in the following format

```
name,short_description
test , test is a person
```

The we create a configuration file `person.json` as follows

```
{
  "source": { "file": { "path": "/home/pi/Documents/entity_person.csv" } },
  "extractor": { "csv": {} },
  "transformers": [
    { "vertex": { "class": "entity_person" } }
  ],
  "loader": {
    "orientdb": {
       "dbURL": "remote:localhost/test2",
	"serverUser": "root",
	"serverPassword": "reloded23",
       "dbType": "graph",
       "classes": [
         {"name": "entity_person", "extends": "V"}
       ], "indexes": [
         {"class":"entity_person", "fields":["name:STRING"], "type":"UNIQUE_HASH_INDEX" }
       ]
    }
  }
}
```

which tell the ETL scripts to take the csv file from specificed location and reate a vertex of entity class
`enity_person`

It also specfies details of loader details on which database to load the data and access details for the same

Finally loader specifies the vertex properties and indexes 

To load the data run the command

```

./oetl.sh person.json
```





**References**

- http://orientdb.com/download/
- http://orientdb.com/docs/2.2/Export-and-Import.html
- http://orientdb.com/docs/2.2/Import-the-Database-of-Beers.html
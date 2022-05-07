---
layout: post
title: Install LetsCrypt SSL Wildcard Certificate on Ubuntu
category: Linux
---

# Install LetsCrypt SSL Wildcard Certificate on Ubuntu

#### **INSTALLING CERTBOT** <a href="#bkmrk-installing-certbot" id="bkmrk-installing-certbot"></a>

```
sudo add-apt-repository ppa:certbot/certbotsudo apt-get updatesudo apt-get install python-certbot-nginx
```

#### **INSTALLING NGINX** <a href="#bkmrk-installing-nginx" id="bkmrk-installing-nginx"></a>

```
sudo apt-get update
sudo apt-get install nginx
```

### Setup DNS to serve all the subdomains <a href="#bkmrk-setup-dns-to-serve-a" id="bkmrk-setup-dns-to-serve-a"></a>

* Create a custom **A** record, HOST **\*** POINTS TO: Your IP Address(Eg: 103.21.0.108)
* Create a custom **A** record, HOST **@** POINTS TO: Your IP Address(Eg: 103.21.0.108)
* Add a CNAME record, HOST **www** POINTS TO **@** this refers to your IP address.

### Obtaining wildcard ssl certificate from Let’s Encrypt <a href="#bkmrk-obtaining-wildcard-s" id="bkmrk-obtaining-wildcard-s"></a>

sudo certbot --server [https://acme-v02.api.letsencrypt.org/directory](https://acme-v02.api.letsencrypt.org/directory) -d \*.**example.com** --manual --preferred-challenges dns-01 certonly

Note:- Replace **example.com** with your domain name

Deploy a **DNS TXT** record provided by Let’s Encrypt certbot after running the above command

### Configuring Nginx to serve wildcard subdomains <a href="#bkmrk-configuring-nginx-to" id="bkmrk-configuring-nginx-to"></a>

* Create a config file `sudo touch /etc/nginx/sites-available/example.com`
* Open the file `sudo vi /etc/nginx/sites-available/example.com`
* Add the following code in the file

```
server {
  listen 80;
  listen [::]:80;
  server_name *.example.com;
  return 301 https://$host$request_uri;
}
server {
  listen 443 ssl;
  server_name *.example.com;
  ssl_certificate /etc/letsencrypt/live/example.com/fullchain.pem;
  ssl_certificate_key /etc/letsencrypt/live/example.com/privkey.pem;
  include /etc/letsencrypt/options-ssl-nginx.conf;
  ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;
  root /var/www/example.com;
  index index.html;
  location / {
    try_files $uri $uri/ =404;
  }
}
```

### Test and restart Nginx <a href="#bkmrk-test-and-restart-ngi" id="bkmrk-test-and-restart-ngi"></a>

* Test Nginx configuration using `sudo nginx -t`
* If it’s success reload Nginx using `sudo /etc/init.d/nginx reload`

Nginx is now setup to handle wildcard subdomains.

**REFERENCES**

[https://medium.com/@utkarsh\_verma/how-to-obtain-a-wildcard-ssl-certificate-from-lets-encrypt-and-setup-nginx-to-use-wildcard-cfb050c8b33f](https://medium.com/@utkarsh\_verma/how-to-obtain-a-wildcard-ssl-certificate-from-lets-encrypt-and-setup-nginx-to-use-wildcard-cfb050c8b33f)

***

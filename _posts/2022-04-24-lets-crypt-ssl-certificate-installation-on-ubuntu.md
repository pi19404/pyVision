# Lets Crypt SSL Certificate Installation on Ubuntu



The first step to using Let’s Encrypt to obtain an SSL certificate is to install the Certbot software on your server.

Certbot is in very active development, so the Certbot packages provided by Ubuntu tend to be outdated. However, the Certbot developers maintain a Ubuntu software repository with up-to-date versions, so we’ll use that repository instead.

First, add the repository.

```
sudo add-apt-repository ppa:certbot/certbot
sudo apt-get update
```

You’ll need to press `ENTER` to accept. Then, update the package list to pick up the new repository’s package information.

And finally, install Certbot’s Nginx package with `apt-get`.

```
sudo apt-get install python-certbot-nginx
```

Certbot is now ready to use, but in order for it to configure SSL for Nginx, we need to verify some of Nginx’s configuration.

Certbot can automatically configure SSL for Nginx, but it needs to be able to find the correct `server` block in your config. It does this by looking for a `server_name` directive that matches the domain you’re requesting a certificate for.

If you’re starting out with a fresh Nginx install, you can update the default config file. Open it with `nano` or your favorite text editor.

Find the existing `server_name` line and replace the underscore, `_`, with your domain name: /etc/nginx/sites-available/default

```
. . .server_name build.miko2.ai;. . .
```

Save the file and quit your editor.

Then, verify the syntax of your configuration edits.

If you get any errors, reopen the file and check for typos, then test it again.

### Obtaining an SSL Certificate with automatic verification <a href="#bkmrk-obtaining-an-ssl-cer" id="bkmrk-obtaining-an-ssl-cer"></a>

Certbot provides a variety of ways to obtain SSL certificates, through various plugins. The Nginx plugin will take care of reconfiguring Nginx and reloading the config whenever necessary:

```
sudo certbot --nginx -d build.miko2.ai
```

This runs `certbot` with the `--nginx` plugin, using `-d` to specify the names we’d like the certificate to be valid for.

If this is your first time running `certbot`, you will be prompted to enter an email address and agree to the terms of service. After doing so, `certbot` will communicate with the Let’s Encrypt server, then run a challenge to verify that you control the domain you’re requesting a certificate for.

If that’s successful, `certbot` will ask how you’d like to configure your HTTPS settings.

```
Please choose whether or not to redirect HTTP traffic to HTTPS, removing HTTP access.
-------------------------------------------------------------------------------
1: No redirect - Make no further changes to the webserver configuration.
2: Redirect - Make all requests redirect to secure HTTPS access. Choose this for
new sites, or if you're confident your site works on HTTPS. You can undo this
change by editing your web server's configuration.
-------------------------------------------------------------------------------
Select the appropriate number [1-2] then [enter] (press 'c' to cancel):
```

Select your choice then hit `ENTER`. The configuration will be updated, and Nginx will reload to pick up the new settings. `certbot` will wrap up with a message telling you the process was successful and where your certificates are stored:

```
IMPORTANT NOTES:
 - Congratulations! Your certificate and chain have been saved at
   /etc/letsencrypt/live/example.com/fullchain.pem. Your cert will
   expire on 2017-10-23. To obtain a new or tweaked version of this
   certificate in the future, simply run certbot again with the
   "certonly" option. To non-interactively renew *all* of your
   certificates, run "certbot renew"
 - Your account credentials have been saved in your Certbot
   configuration directory at /etc/letsencrypt. You should make a
   secure backup of this folder now. This configuration directory will
   also contain certificates and private keys obtained by Certbot so
   making regular backups of this folder is ideal.
 - If you like Certbot, please consider supporting our work by:

   Donating to ISRG / Let's Encrypt:   https://letsencrypt.org/donate
   Donating to EFF:                    https://eff.org/donate-le
```

Your certificates are downloaded, installed, and loaded



### Obtaining  ssl certificate from Let’s Encrypt with manual verification <a href="#bkmrk-obtaining-wildcard-s" id="bkmrk-obtaining-wildcard-s"></a>

sudo certbot --server [https://acme-v02.api.letsencrypt.org/directory](https://acme-v02.api.letsencrypt.org/directory) -d **example.com** --manual --preferred-challenges dns-01 certonly

Note:- Replace **example.com** with your domain name

Deploy a **DNS TXT** record provided by Let’s Encrypt certbot after running the above command



REFERENCES

* [https://www.digitalocean.com/community/tutorials/how-to-secure-nginx-with-let-s-encrypt-on-ubuntu-16-04](https://www.digitalocean.com/community/tutorials/how-to-secure-nginx-with-let-s-encrypt-on-ubuntu-16-04)

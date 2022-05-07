# Install ngrok

create an account and login in to ngrok

Download the linux package

```
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz --no-check-certificate
tar -zxvf ngrok-v3-stable-linux-amd64.tgz
mv ngrok /usr/bin/ngrok
chmod 755 /usr/bin/ngrok
```



Running this command will add your authtoken to the default `ngrok.yml` configuration file. This will grant you access to more features and longer session times. Running tunnels will be listed on the [endpoints page](https://dashboard.ngrok.com/cloud-edge/endpoints) of the dashboard.

```
ngrok config add-authtoken {TOKEN}
```



o start a HTTP tunnel forwarding to your local port 80, run this next:

```
ngrok http 8800
```



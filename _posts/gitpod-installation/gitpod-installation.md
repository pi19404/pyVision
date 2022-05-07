# GitPod Installation



To start with installing Gitpod, you need a terminal where you can run `kubectl` to access your cluster. At first, install the KOTS kubectl plugicurl https://kots.io/install | bash



```shell
kubectl kots install gitpod
```

You will be asked for the namespace you want to install Gitpod to as well as a password for the admin console. After some time, you will see the following output:

```
  • Press Ctrl+C to exit
  • Go to http://localhost:8800 to access the Admin Console
```

To access the Admin Console again, run kubectl kots admin-console --namespace gitpod



Open your favorite browser and go to `http://localhost:8800` (port `8800` is opened on your node on `localhost` only—you may want to forward the port to your workstation in order to access the admin console).

we will use ngrok for port forwarding

```
ngrok http 8800
```

![](<../../.gitbook/assets/image (6) (1).png>)

The first page will ask you to upload your Gitpod license. Gitpod provides community license for free (right click and save link as [here](https://raw.githubusercontent.com/gitpod-io/gitpod/main/install/licenses/Community.yaml))

.&#x20;

![](<../../.gitbook/assets/image (1).png>)

Enter the domain name as :gitpod.miko-robot.co.in

User all the default settings for incluster container registry , mysql database , storage provide

User cert-manager for SSL certificates and use Issuer type as "Issuer"

Mark Allow login by into workspace via ssh



Check if all the pods are running states

![](<../../.gitbook/assets/image (7).png>)



```
kubectl -n gitpod patch svc proxy -p '{"spec": {"type": "LoadBalancer", "externalIPs":["164.52.207.58"]}}'
```

Once the installation has been finished successfully, you will see the status “Ready” with a small green indicator next to the Gitpod logo. You see which version you installed and which license you are using.

![](<../../.gitbook/assets/image (6).png>)

# Install cert-manager on kubernetes cluster

cert-manager is a Kubernetes add-on to automate the management and issuance of TLS certificates from various issuing sources.

It will ensure certificates are valid and up to date periodically, and attempt to renew certificates at an appropriate time before expiry.

on the same node where k3s master was installed run the below command to install the cert manager

```
k3s kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.8.0/cert-manager.yaml
    
```

By default, cert-manager will be installed into the `cert-manager` namespace

**Creating TLS certs for your domain with cert-manager**

cert-manager provides the Gitpod installation with certificates for internal communication. Besides this, cert-manager can also create a TLS certificate for your domain. Since Gitpod needs wildcard certificates, you must use the [DNS-01 challenge](https://letsencrypt.org/docs/challenge-types/#dns-01-challenge).

You can use either an [`Issuer` or `ClusterIssuer`](https://cert-manager.io/docs/concepts/issuer).

`Issuers`, and `ClusterIssuers`, are Kubernetes resources that represent certificate authorities (CAs) that are able to generate signed certificates by honoring certificate signing requests. All cert-manager certificates require a referenced issuer that is in a ready condition to attempt to honor the request.

**we will use the domain** miko-robot.co.in , DNS provide is onlydomains . We will first delegate the domain to cloudflare and then use cloudflare for DNS01 challenge&#x20;

Create and Account and Login into cloudflare . Click on Add a Site option to start the domain delegation process

![](<{{ site.baseurl }}/images/gitbook/assets/image (2).png>)

For now choose the Free option and continue

![]({{ site.baseurl }}/images/gitbook/assets/image.png)

**Log in** to the **administrator account** for your domain registrar . In this case domain registrar is onlydomain.com .&#x20;

1.  **By Default** the following nameservers are configured

    ```
    ns2.onlydomains.com , ns3.onlydomains.com , ns1.onlydomains.com
    ```


2. **Go to DNS Settings Menu**

**Choose the option to delegate name server Add** Cloudflare's nameservers

```
tegan.ns.cloudflare.com
```

```
zod.ns.cloudflare.com
```

![](<{{ site.baseurl }}/images/gitbook/assets/image (4).png>)

This will update the DNS Settings and allow the DNS to be managed via cloudflare

### Create CloudFlare API Tokens <a href="#api-tokens" id="api-tokens"></a>

Tokens can be created at **User Profile > API Tokens > API Tokens**. The following settings are recommended:

* Permissions:
  * `Zone - DNS - Edit`
  * `Zone - Zone - Read`
* Zone Resources:
  * `Include - All Zones`

![](<{{ site.baseurl }}/images/gitbook/assets/image (5).png>)

Copy The token and save it as it will not be displayed again for security purposes

**Verify that the token is working**

```
curl -X GET "https://api.cloudflare.com/client/v4/user/tokens/verify"
-H "Authorization: Bearer {TOKEN}"
-H "Content-Type:application/json"
```

If token is working then you will see a output similar to one below

```
{"result":{"id":"cf4a06f05d43d58468667ba715145c34","status":"active"},"success":true,"errors":[],"messages":[{"code":10000,"message":"This API Token is valid and active","type":null}]}
```

**Create a new Issuer**

To create a new `Issuer`, first make a Kubernetes secret containing your new API token:

```
apiVersion: v1
kind: Secret
metadata:
 name: cloudflare-api-token-secret
 namespace: cert-manager
type: Opaque
stringData:
 api-token: {TOKEN}
```

Please note that the namespace has to be cert-manager for the key else you may encounter error while creating the certificates

To apply the configuration

```
k3s kubectl apply -f cloudflare_token.yaml
```

**Create Issuer configuration file**

```
kind: ClusterIssuer
metadata:
  name: gitpod-issuer
spec:
  acme:
    email: prashant@miko.ai
    server: https://acme-staging-v02.api.letsencrypt.org/directory
    privateKeySecretRef:
      name: gitpod-issuer
    solvers:
    - dns01:
        cloudflare:
          email : aaa@aaa.com
          apiTokenSecretRef:
            name: cloudflare-api-token-secret
            key: api-token
```

Once you complete the gitpod installation create the below certificates in gitpod and kube-system workspace both

```
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
 name: https-certificates
 namespace : kube-system
spec:
 secretName: https-certificates
 issuerRef:
   name: gitpod-issuer
   kind: ClusterIssuer
 dnsNames:
  - gitpod.miko-robot.co.in
  - "*.gitpod.miko-robot.co.in"
  - "*.ws.gitpod.miko-robot.co.in"
```

```
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
 name: https-certificates
 namespace : gitpod
spec:
 secretName: https-certificates
 issuerRef:
   name: gitpod-issuer
   kind: ClusterIssuer
 dnsNames:
  - gitpod.miko-robot.co.in
  - "*.gitpod.miko-robot.co.in"
  - "*.ws.gitpod.miko-robot.co.in"
```

To apply the configuration

```
k3s kubectl apply -f cert.yaml
```

While certificate issuance process is in progress you will see the status as False

```
kubectl get certificate
NAME                        READY   SECRET                      AGE
https-certificates          False    https-certificates          5m
```

After a few minutes, you should see the `https-certificate` become ready.

```
kubectl get certificate
NAME                        READY   SECRET                      AGE
https-certificates          True    https-certificates          5m
```



Once the DNS record has been updated, you can delete all Cert Manager pods to retrigger the certificate request

```
kubectl delete pods -n cert-manager --all
```

**References**

[https://cert-manager.io/docs/configuration/acme/dns01/cloudflare/](https://cert-manager.io/docs/configuration/acme/dns01/cloudflare/)




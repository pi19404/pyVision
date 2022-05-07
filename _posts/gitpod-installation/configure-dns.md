# Configure DNS

To install Gitpod you need a domain with a TLS certificate. The DNS setup to your domain needs to be configured such that it points to the ingress of your Kubernetes cluster. You need to configure your actual domain (say `example.com`) as well as the wildcard subdomains `*.example.com` as well as `*.ws.example.com`.

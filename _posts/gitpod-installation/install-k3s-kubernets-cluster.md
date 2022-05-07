# Install K3S Kubernets Cluster

## Cluster Set-Up <a href="#cluster-set-up" id="cluster-set-up"></a>

Gitpod is a Kubernetes application running with certain expectations on the characteristics of the cluster it is running on.

Gitpod requires Kubernetes as an orchestration technology in order to spin up and down workspaces—ideally in combination with cluster autoscaling to minimise cost. We strongly recommend deploying a dedicated Kubernetes cluster just for Gitpod Self-Hosted.

In this article we will use k8s to setup a self managed kubernetes cluster

K3s is a highly available, certified Kubernetes distribution designed for production workloads in unattended, resource-constrained, remote locations or inside IoT appliances.



On each node, we [install K3s](https://rancher.com/docs/k3s/latest/en/installation/). We configure K3s by setting the following environment variables on the nodes.

K3s config for main node `node0`:&#x20;

The below configure is for a single node setup with only master node

```shell
export INSTALL_K3S_EXEC="server --disable traefik --flannel-backend=none --node-label gitpod.io/workload_meta=true --node-label gitpod.io/workload_ide=true -node-label gitpod.io/workload_workspace_services=true --node-label gitpod.io/workload_workspace_regular=true --node-label gitpod.io/workload_workspace_headless=true"
export K3S_CLUSTER_SECRET="44dc0c2d471f50bc151aa72515d53067"
curl -sfL https://get.k3s.io | sh -
```

After setting the environment variables, install K3s on every node like this:

```shell
$ curl -sfL https://get.k3s.io | sh -
```

```
# Check for Ready node,
takes maybe 30 seconds
k3s kubectl get node
```

You can run the below command to start the server&#x20;

```
systemctl start k3s.service
# Kubeconfig is written to /etc/rancher/k3s/k3s.yaml
k3s kubectl get node
```

To enable any other node to join the cluster run the command

```
# On a different node run the below. NODE_TOKEN comes from /var/lib/rancher/k3s/server/node-token
# on your server
export INSTALL_K3S_EXEC="agent --node-label gitpod.io/workload_workspace_services=true --node-label gitpod.io/workload_workspace_regular=true --node-label gitpod.io/workload_workspace_headless=true"
export K3S_CLUSTER_SECRET="<your random secret string that is the same on all nodes>"
export K3S_URL="https://node0:6443"
```

Now, you have to install [Calico](https://www.tigera.io/project-calico/).&#x20;

**Method 1**

Download the [Calico manifest](https://docs.projectcalico.org/manifests/calico-vxlan.yaml) and add the following line to the `plugins` section of the `cni_network_config`:

```json
"container_settings": { "allow_ip_forwarding": true }
```

```
/var/lib/rancher/k3s/server/manifests/
```

Copy that file to `node0` in the following folder (create folder if missing):

**Method 2**



Install the Calico operator and custom resource definitions.

```
kubectl create -f https://projectcalico.docs.tigera.io/manifests/tigera-operator.yaml
```

Install Calico by creating the necessary custom resource. For more information on configuration options available in this manifest, see [the installation reference](https://projectcalico.docs.tigera.io/reference/installation/api).

```
kubectl create -f https://projectcalico.docs.tigera.io/manifests/custom-resources.yaml
```

> **Note**: Before creating this manifest, read its contents and make sure its settings are correct for your environment. For example, you may need to change the default IP pool CIDR to match your pod network CIDR.

**Final checks**

1. Confirm that all of the pods are running using the following command.

```
watch kubectl get pods --all-namespaces
```

Wait until each pod shows the `STATUS` of `Running`.

![](<{{ site.baseurl }}/images/gitbook/assets/image (7) (1).png>)

Confirm that you now have a node in your cluster with the following command.

```
kubectl get nodes -o wide
```

![](<{{ site.baseurl }}/images/gitbook/assets/image (8).png>)\
References

* [https://www.gitpod.io/docs/self-hosted/latest/cluster-set-up/on-k3s](https://www.gitpod.io/docs/self-hosted/latest/cluster-set-up/on-k3s)
* [https://www.gitpod.io/docs/self-hosted/latest/getting-started#step-2-install-cert-manager](https://www.gitpod.io/docs/self-hosted/latest/getting-started#step-2-install-cert-manager)
* [https://projectcalico.docs.tigera.io/getting-started/kubernetes/k3s/quickstart](https://projectcalico.docs.tigera.io/getting-started/kubernetes/k3s/quickstart)

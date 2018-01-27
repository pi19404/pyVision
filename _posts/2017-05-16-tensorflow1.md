---
layout: post
title: NLP Installing TensorFlow on ubuntu 16.04
published: true
---


[TensorFlow](https://www.tensorflow.org/) is An open-source software library for Machine Intelligence.
In this article will be installing tensorflow on ubuntu 16.04 with GPU support.

# **CUDA Installation **


**Verify You Have a CUDA-Capable GPU**

To verify that your GPU is CUDA-capable, at the command line, enter:

```
pi@pi-dektop:~$ lspci | grep -i nvidia
01:00.0 VGA compatible controller: NVIDIA Corporation GK208 [GeForce GT 730] (rev a1)
01:00.1 Audio device: NVIDIA Corporation GK208 HDMI/DP Audio Controller (rev a1)
```

If your graphics card is from NVIDIA and it is listed in http://developer.nvidia.com/cuda-gpus, your GPU is CUDA-capable.

**Verify You Have a Supported Version of Linux**

The CUDA Development Tools are only supported on some specific distributions of Linux. These are listed in the CUDA Toolkit release notes.

To determine which distribution and release number you're running, type the following at the command line:

```
$ uname -m && cat /etc/*release

x86_64
DISTRIB_ID=Ubuntu
DISTRIB_RELEASE=16.04
DISTRIB_CODENAME=xenial
DISTRIB_DESCRIPTION="Ubuntu 16.04.1 LTS"
NAME="Ubuntu"
VERSION="16.04.1 LTS (Xenial Xerus)"
ID=ubuntu
ID_LIKE=debian
PRETTY_NAME="Ubuntu 16.04.1 LTS"
VERSION_ID="16.04"
HOME_URL="http://www.ubuntu.com/"
SUPPORT_URL="http://help.ubuntu.com/"
BUG_REPORT_URL="http://bugs.launchpad.net/ubuntu/"
UBUNTU_CODENAME=xenial
```

The x86_64 line indicates you are running on a 64-bit system.

**Verify the System Has gcc Installed**

The gcc compiler is required for development using the CUDA Toolkit. It is not required for running CUDA applications. It is generally installed as part of the Linux installation, and in most cases the version of gcc installed with a supported version of Linux will work correctly.

To verify the version of gcc installed on your system, type the following on the command line:

```
pi@pi-dektop:~$  gcc --version
gcc (Ubuntu 5.4.0-6ubuntu1~16.04.4) 5.4.0 20160609
Copyright (C) 2015 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

If an error message displays, you need to install the development tools from your Linux distribution or obtain a version of gcc and its accompanying toolchain from the Web.

**Verify the System has the Correct Kernel Headers and Development Packages Installed**

The CUDA Driver requires that the kernel headers and development packages for the running version of the kernel be installed at the time of the driver installation

The version of the kernel your system is running can be found by running the following command:

```
pi@pi-dektop:~$ uname -r
4.4.0-75-generic
```

The kernel headers and development packages for the currently running kernel can be installed with:

```
$ sudo apt-get install linux-headers-$(uname -r)
```

**Download CUDA**

CUDA can be downloaded from https://developer.nvidia.com/cuda-downloads

Select the relevant operating system,architecture type,OS distribution and version to download the appropriate file

we will be installing cuda toolking using standalone installed which downloads the `.run` file by choosing the `runfile(local)` option on the download screen.Optionally you can choose other options like to download a `.deb` local or network package for installation.

Run `sudo sh cuda_8.0.61_375.26_linux.run` and follow the instruction to install the latest nvidia drivers

You have have to enter console mode and stop the desktop environment to install nvidia drivers

- Hit CTRL+ALT+F1 and login using your credentials.
- kill your current X server session by typing sudo service lightdm stop or sudo lightdm stop
- `sudo sh cuda_8.0.61_375.26_linux.run` and follow installation instructions
- sudo reboot

You may have to reboot the PC after installation.

Upon restart if you face any issue run the command

```
sudo apt-get install nvidia-current
```

**Download NVIDIA Drivers**


```
sudo apt-get install -y nvidia-367 nvidia-prime
sudo update-alternatives --config x86_64-linux-gnu_gl_conf
LD_PRELOAD=/usr/lib/nvidia-375/libnvidia-ml.so nvidia-smi
sudo apt-get install libcupti-dev
```

# **TENSORFLOW INSTALLATION**

## **Virtualenv**

 Virtualenv is a virtual Python environment isolated from other Python development, incapable of interfering with or being affected by other Python programs on the same machine. During the virtualenv installation process, you will install not only TensorFlow but also all the packages that TensorFlow requires
 
 To start working with TensorFlow, you simply need to "activate" the virtual environment. All in all, virtualenv provides a safe and reliable mechanism for installing and running TensorFlow.
 
 **Install pip and virtualenv by issuing the following command:**
 
 ```
$sudo apt-get install python-pip python-dev python-virtualenv 
 $sudo pip install virtualenv virtualenvwrapper
```

**Create a virtualenv environment by issuing the following command:**

```
virtualenv --system-site-packages targetDirectory 
mkvirtualenv tensorflow
```

**Activate the virtualenv environment by issuing one of the following commands:**

```
workon tensorflow
source /opt/python/virtualenv/bin/activate
```

Note that you must activate the virtualenv environment each time you use TensorFlow

When you are done using TensorFlow, you may deactivate the environment by invoking the deactivate

```
deactivate
```

**Install tensorflow**

we will install tensorflow and supporting libraries required for running machine learning algorithms

```
pip install --upgrade tensorflow-gpu
sudo pip install keras
sudo pip install scikit-learn gensim pandas ijson nltk sklearn spacy
python -m spacy download en

```

**Install NVIDIA Machine learning softwares**

NVIDIA requirements to run TensorFlow with GPU support
- CUDA® Toolkit 8.0
- The NVIDIA drivers associated with CUDA Toolkit 8.0
- cuDNN v5.1
- libcupti-dev library, which is the NVIDIA CUDA Profile Tools Interface

Download the  NVIDIA [cuDNN](https://developer.nvidia.com/rdp/cudnn-download) is a GPU-accelerated library of primitives for deep neural networks.

We will be using the cuDNN Version 5 which is compatible with cuda8.0
Select the option `cuDNN v5 Library for Linux  Ubuntu16.04 (Deb)` and follow the instructions to install the file

```
gunzip cudnn-8.0-linux-x64-v5.0-ga.tgz
tar -xvf cudnn-8.0-linux-x64-v5.0-ga.tar
cp -rf cuda /usr/local/cuda/
```

**Check Tensorflow installation**

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/lib/nvidia-381/
python -c "import tensorflow; print(tensorflow.__version__)"
```

If you get output then tensorflow has been installed properly

**References**

- https://www.tensorflow.org/install/
- http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#axzz4hDncLYTs
- http://exponential.io/blog/2015/02/10/install-virtualenv-and-virtualenvwrapper-on-ubuntu/
- http://exponential.io/blog/2015/02/10/configure-pycharm-to-use-virtualenv/
- https://keras.io/#installation
- https://developer.nvidia.com/cuda-downloads
- https://www.tensorflow.org/install/install_linux
- http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#runfile-nouveau
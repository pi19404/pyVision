---
published: false
---




**Introduction**
In this article we look at process of benchmkarking ARM based processors using CoreMark

**CoreMark**
CoreMark is a modern, sophisticated benchmark that lets you accurately measure processor performance.

Developed in 2009 by the Embedded Microprocessor Benchmark Consortium (EEMBC),
CoreMark is a freely available, easily portable benchmark program that measures processor performance

**Pre requisites**

We need to install some softwares that enable us to compile and run code on windows7 system.

- **GNU Tools for ARM Embedded Processors**
	You can download the ARM toolchains from https://launchpad.net/gcc-arm-embedded/+download

- **COOCOX IDE **
	We will use [CoIDE](http://www.coocox.org/software.html) is a free integrated development environment focusing on ARM Cortex-M0/M0+/M3/M4 based microcontrollers

- **COREMARK Sources **
The official CoreMark source is available from the CoreMark web site,
http://www.coremark.org.

- **MBED SDK **
The [mbed Software Development Ki](https://developer.mbed.org/handbook/mbed-SDK)t (SDK) is a C/C++ microcontroller opensource software platform built on the low-level ARM CMSIS APIs enabling fast development and deployment.

The MBED sources can be downloaded from github repository https://github.com/mbedmicro/mbed

To compile the SDK we require some software packages like
- [python](https://www.python.org/downloads/windows/)
- git


we will use precompiled MBED libraries in our project.

The official mbed C/C++ SDK provides the software platform and libraries to build your applications.

We be running benchmarks on the following development platform boards
- Nucleo F334R8 - 72MHz ARM Cortex M4 
- Nucleo F303R3 - 72MHz ARM Cortex M4
- Nucleo F091RC - 48MHz ARM Cortex M0
- STM32F Discovery board - 

MBED sources can be built using online compiler https://developer.mbed.org/compiler/
Go to the https://developer.mbed.org/platforms/ and find the platforms corresponding to the nucleo boards and click on `Add to Your mbed compiler` option to include the platform in the user configuration.

This will enable us to download precompiled libraries and files relevant to the platform

The MBED header and library files for various platforms can be found at
https://developer.mbed.org/users/mbed_official/code/mbed/

Select the platform and Click on the option "Export to Desktop IDE" to download
the files relevant to the SDK.

Choose the platform as `Nucleo boards` and toolchains as `ARM GCC`




https://developer.mbed.org/questions/5371/CooCox-build-of-exported-blink_led-examp/



we will  build and run CoreMark on bare-metal systems and compare the performances of a few ARM based microcontrollers.


Compiling CoreMark with ARM Compiler

CoreMark consists of the following C source and header files:
• coremark.h
• core_main.c
• core_list_join.c
• core_matrix.c
• core_state.c
• core_util.c
• simple/core_portme.c
• simple/core_portme.h

The source files in the `simple` directory need to be modified to target CoreMark to your particular platform.


Library functions required by CoreMark








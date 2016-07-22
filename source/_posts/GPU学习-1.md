---
title: GPU学习
date: 2016-07-22 16:51:59
categories:
- GPU
tags:
- GPU
---

实验环境：
系统是Ubuntu，cuda的版本是7.5。

具体怎么搭建还是参考官方文档[documents](http://docs.nvidia.com/cuda/index.html#axzz4DzlUApap)

## 1、Introduction

### 1、1 From Graphics Processing to General Purpose Parallel Computing

由于市场对实时性、3D高清图像的无限制需求的驱动，可编程的GPU(Graphic Processor Unit)已经发展成为一个高度并行的、多线程、多core的处理器。拥有大计算功率和高网络带宽。如下图所示：

图1对比了cpu和GPU每秒的浮点运算，图2对比了cpu和GPU的网络带宽。

![Figure 1](http://docs.nvidia.com/cuda/cuda-c-programming-guide/graphics/floating-point-operations-per-second.png)

![Figure 2](http://docs.nvidia.com/cuda/cuda-c-programming-guide/graphics/memory-bandwidth.png)

CPU和GPU的浮点运算的差异性原因在于GPU是专门用于计算密集型的(compute-intensive)和高度并行的计算，这也是图像渲染的特点。这样的设计可以使用更多的晶体管来专注于数据处理而不是数据缓存和流程控制。如图3所示：

![Figure 3](http://docs.nvidia.com/cuda/cuda-c-programming-guide/graphics/gpu-devotes-more-transistors-to-data-processing.png)

更特殊的是，GPU是更适用于处理能够数据并行计算的问题。同样的程序可以并行的在多个数据单元上(data element)执行，这样的程序一般是高计算密度，也就是计算操作和内存操作的比值比价大。

因为每个数据单元执行同样的程序，所以对于复杂的流程控制操作并不太需要。因为计算密度比较高，所以内存访问的延迟会隐藏在计算中而不是大数据缓存中。

数据并行操作把数据单元映射到并行处理的线程中。许多处理大数据集的应用都可以了使用数据并行编程模型来加速计算。在3D模型渲染中，大量的像素点集被映射到并行线程中。同样的，图像和视频处理的应用中，比如渲染图像的后期处理，视频的编解码，图像缩放，立体视觉和模式识别中可以把图像块和像素映射到并行处理线程中。实际上，在图像渲染和处理领域外的很多算法都可以用数据并行来处理。比如一般的信号处理和物理模拟计算金融学和计算生物学。

### 1.2 CUDA：A General-Purpose Parallel Computing Platform and Programming Model

在2006年11月份，NVIDIA引入了CUDA，一个通用的并行计算框架和编程模型来提升NVIDIA的GPU的并行计算引擎，用来解决许多复杂的计算问题并且比CPU更有效。

CUDA是一个允许开发者使用C语言或者更高级的语言的软件开发环境，如下图4所示，别的语言，别的应用编程接口或者直接的方法调用都是支持的。

![Figure 4](http://docs.nvidia.com/cuda/cuda-c-programming-guide/graphics/gpu-computing-applications.png)

### 1.3 A Scalable Programming Model

多核CPU和GPU的到来意味着主流处理芯片都是并行处理系统。更远的说，这也是扩展了摩尔定律。
现在的挑战是开发应用软件来显式的扩展并发度，来利用不断增长处理核心数量。比如3D图像应用利用多核GPU显式的提升了并行度。

CUDA并行模型的设计就是用来给哪些熟悉标准的编程语言比如C的编程人员很低的学习曲线。

CUDA内部的核心有三个概念：
* 分层的线程组(a hierarchy of thread groups)
* 共享内存( shared memories)
* 栅栏同步(barrier synchronization)

这样暴露给编程人员的是一个很小的语言扩展集合。

这几个抽象概念提供了把细粒度的数据并行和线程并行嵌入到粗粒度的数据并行和任务并行。我们会引导程序员把问题分割成粗粒度的子任务，每个子任务都可以独立并行的在线程块中运行，并且每个子任务都可以在所有的并发的线程块中合作的解决。

这样的分解保留了语言的表达力，通过允许线程合作的解决子任务，同时可以自动扩展。
确实的，线程中的每一个块都是可以通过多核GPU来调度，调度的顺序可以是任何顺序，比如并发的，比如顺序的。那么一个编译好的CUDA程序可以执行在任何数量的多核处理器，如图5所示，而且只有运行时系统才需要知道物理处理器核心的数量。

![Figure 5](http://docs.nvidia.com/cuda/cuda-c-programming-guide/graphics/automatic-scalability.png)

GPU建立在一组多处理器（SMX，Streaming Multiprocessors）附近。
一个SMX的配置：
* 192 cores（都是SIMT cores（Single Instruction Multiple Threads） and 64k registers,GPU中的SIMT对应于CPU中的SIMD（Single Instruction Multiple Data）
* 64KB of shared memory / L1 cache
* 8KB cache for constants
* 48KB texture cache for read-only arrays
* up to 2K threads per SMX


每个multi-thread程序的execution kernel instance（kernel定义见下一节,instance指block）在一个SMX上执行，一个多线程程序会分配到blocks of threads（每个block中负责一部分线程）中独立执行。所以GPU中的处理器越多执行越快（因为如果SMX不够给每个kernel instance分配一个，就要几个kernel抢一个SMX了）。具体来讲，如果SMX上有足够寄存器和内存（后面会讲到，shared memory），就多个kernel instance在一个SMX上执行，否则放到队列里等。

GPU工作原理：首先通过主接口读取中央处理器指令，GigaThread引擎从系统内存中获取特定的数据并拷贝到显存中，为显存控制器提供数据存取所需的高带宽。GigaThread引擎随后为各个SMX创建和分派线程块（warp, 详细介绍见SIMT架构或者CUDA系列学习（二）），SMX则将多个Warp调度到各CUDA核心以及其他执行单元。在图形流水线出现工作超载的时候，GigaThread引擎还负责进行工作的重新分配。

## 2、Programming Model

这一节主要介绍CUDA编程模型背后的主要概念。用C语言来描述，更多的CUDA c接口扩展描述可以参考[Programming Interface](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-interface)

### 2.1 Kernels

CUDA C 通过运行程序员定义C的函数，这样就扩展了C，我们称之为Kernels, Kernels被调用时
就会在N个不同的CUDA线程中执行N次(一个CUDA线程执行一次，那么一共就是次)。普通的C函数都是执行一次。

kernel用__global__这个声明符号来定义，CUDA线程执行给定的kernel的调用的数据通过new<<<...>>>来指定(see [C Language Extensions](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#c-language-extensions))。

每个执行kernel的线程给定一个唯一的线程ID，可以通过在kernel中内置的threaIdx变量来获取。

接下来的代码求两个长度为N的向量的和，并且把结果存储在向量C中。


      // Kernel definition
      __global__ void VecAdd(float* A, float* B, float* C)
      { int i = threadIdx.x;
        C[i] = A[i] + B[i];
      }
      int main()
      {
        ...
        // Kernel invocation with N threads
        VecAdd<<<1, N>>>(A, B, C);
        ...
      }

这里N个线程中的每个都执行VecAdd()，并且是向量中的其中一对的和。

### 2.2 Thread Hierarchy

为了方便起见，threadIdx是一个3维(3-component)的向量,那么线程就可以用一维、二维、三维的线程索引来标记，组成了一维、二维、三维的线程块。这样就提供了一个自然的方式来计算向量、矩阵、卷的元素。

线程的索引和线程的ID彼此之间相关：对于一个一维的块，他们是一样的，对于两维的大小是(Dx,Dy)的块,线程索引(x,y)的线程ID就是(x + yDx);对于三维的大小是(Dx,Dy,Dz)的块 ，线程索引(x,y,z)的线程ID就是(x + yDx + zDxDy)。

下面的例子就是相加大小是N*N的矩阵A和矩阵B，然后存入矩阵C。

    // Kernel definition
    __global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N])
    {
      int i = threadIdx.x;
      int j = threadIdx.y;
      C[i][j] = A[i][j] + B[i][j];
      } int main()
      {
        ... // Kernel invocation with one block of N * N * 1 threads
        int numBlocks = 1;
        dim3 threadsPerBlock(N, N);
        MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
        ...
      }

对于每个块的线程数有个限制，因为一个块的所有线程应该都在同一个处理器核，并且共享处理器核上的有限的内存资源。在当前的GPU，一个线程块可以包含1024个线程。

不管怎么样，一个kernel可以在多个等大小(equally-shaped)的线程块中执行，所以线程的总量等于每个线程块中的线程总量*线程块数量。

线程块可以组成一维、二维、三维的grid，如图六所示。在grid中的线程块数量一般由需要处理的数据的大小或者是系统处理器数量决定的。

![Figure 6](http://docs.nvidia.com/cuda/cuda-c-programming-guide/graphics/grid-of-thread-blocks.png)

每个块的线程数量和grid中块的数量由<<<...>>>这样的语法指定，可以是int，或者dim3。二维的块或者grid在上面的例子代码有说明。

grid中的每个块可以由一维、二维、三维的索引指定，这些索引是在kernel中通过内置的blockIdx获取，线程块的纬度可以在kernel中通过内置的blockDim变量获取。

扩展前面的MatAdd()例子来处理多个块，代码如下：

    // Kernel definition
    __global__ void MatAdd(float A[N][N], float B[N][N],
      float C[N][N])
      {     
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        if (i < N && j < N)
        C[i][j] = A[i][j] + B[i][j];
      }

      int main()
      {
        ...
        // Kernel invocation
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
        MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
        ...
      }

一个16*16大小的线程块(256个线程)是一个正常的选择,虽然在这个例子中是任意的大小。足够的线程块，就可以每个每个矩阵元素分配一个线程，这样就可以创建一个grid。

简单的说，这个例子假设每个维度中的grid中的线程数量最终分割成同一个维度的线程块中的线程数，虽然在这里例子中并不是这样的。

线程块需要独立的执行：它们可以任意顺序的执行，并行或者按顺序。独立性也就允许了线程块可以在不同的处理器核上按任意顺序调度。如图5所示，这样也允许了程序员可以写在不同的处理核上可以扩展的代码。

同一个线程块中的线程中的线程可以通过一些共享的内存中的数据来合作处理，也可以协调线程间的内存访问来同步线程。更精确的说你可以在kernel代码中调用__syncthreads()这个内部函数来同步操作。\__syncthreads()的功能就像是栅栏(barrier),也就是说线程块中的所有线程除非被允许执行，否则就一直等待下去。[ Shared Memory](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)给出了如何使用共享内存的例子。

为了有限的合作，共享内存必须是每个处理器的低延迟区域(比如L1缓存),而且__syncthreads()是轻量级的操作。

### Memory Hierarchy

CUDA的线程可以在执行过程中从不同的内存空间中访问数据，如图7所示。每个线程都有自己的私有本地内存。每个线程块都有自己内部线程可见的共享内存，并且伴随着线程块的生命周期。所有的线程都可以访问同样的全局内存。

还有两个额外的所有线程都可以访问的只读内存空间：常量区(the constant) 和纹理内存空间(texture memory spaces)。全局、常量区、纹理内存空间都是为不同的用处优化过的(see [ Device Memory Accesses](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses)。纹理内存还提供了不同的访问模式，就跟数据过滤一样，一些具体的数据格式可以参考[Texture and Surface Memory](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-and-surface-memory)。

内存全局区、常量区、纹理区空间在应用程序的kernel运行时就持久化了。

![Figure 7](http://docs.nvidia.com/cuda/cuda-c-programming-guide/graphics/memory-hierarchy.png)

### Heterogeneous Programming

就像图8描述的那样，CUDA编程模型假设CUDA线程在物理上独立的设备上执行，就像是在主机上的协处理器上执行c程序一样。在这个例子中，GPU只执行kernel部分，其它c语言代码在CPU上执行。

CUDA编程模型假设主机(host)和设备(device)维护了在DRAM中自己独立的空间，也就是说主机内存和设备内存是相互独立的。因为一个CUDA的程序管理全局内存、常量区内存、纹理区内存，通过调用CUDA runtime让它们对kernels可见，见[Programming Interface](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-interface)。这也包含了设备内存数据在主机和设备上转移时内存的分配和释放。

![Figure 8](http://docs.nvidia.com/cuda/cuda-c-programming-guide/graphics/heterogeneous-programming.png)

### Compute Capability

一块GPU设备的计算能力是由版本号决定的，有时又称之为SM version。这个版本号标识了GPU设备能够支持的特性，并且应用程序在运行时来决定当前GPU硬件哪些特性或者指令被使用。

计算能力由一个主修订号X和一个最小的修订号组成，写成：X.Y。

拥有同样的主修订号的设备有同样的核心架构。比如主修订号5表示设备是基于Maxwell架构，3表示Kepler架构，2表示Fermi架构，1表示Tesla架构。

最小的修订号表示在这个核心架构上的一些持续改进，包含了一些新的特性。

[CUDA-Enabled GPUs](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-enabled-gpus)列出了所有的支持CUDA的设备和它们的计算能力。[ Compute Capabilities](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)给出了计算能力的技术说明。

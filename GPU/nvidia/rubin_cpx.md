#  Rubin CPX 

> [原文地址](https://mp.weixin.qq.com/s/5oyNhocptR0PD4IB3p5wXA)

总体来看, Rubin CPX 只是原来 Hopper 那一代 L40 和 Blackwell 这一代的 RTX6000 pro 这条产品线的延续, 然后重新包装成了一个 Context GPU 的概念并且集成到 Rubin NVL 144中搭售。也就是说 Rubin CPX 并不是专门为 Long Context 设计的, 而是恰好适合来做这件事情. 另一方面我们也看到了 RTX 6000 pro 的销售似乎并不好, 而 GB200 NVL72 相对于 B200 的 ROI 也并不是那么的好, 贵了1.5倍? 然后性能只有少数一些case下才有40%以上的收益. 实际上 B200 更划算? NVL的故事如何讲呢?

官方的两种方案, VR CPX NVL144 采用固定配比, 并且丧失了 ScaleOut GDR 的能力, 而 Dual-Rack 方案虽然天然的支持 xPyD, 但又会导致<span style="color:red">在 ScaleOut 上同时进行KVCache 传输和 EP 并行的流量产生干扰的问题</span>, 同时多了很多 Vera CPU 和 CX9 网卡. 老黄那句“Buy More, save More”是否成立?

如果专门For Long Context那么为啥不做一个带两个PCIe Gen6x16的 Rubin CPX呢? 进一步扩大ScaleOut带宽?

当然另外一方面, 模型本身也在尝试 MoR 和 Universal Transformer 以及一些Linear Attn的事情. 前面两个对于 Attn 计算的算力要求会更高, 而后面Linear Attn对算力要求会低不少. 这些Trade-off如何处理呢? 等后面有时间从数学上详细分析一下Linear Attn再说吧.

总体来看, Rubin CPX可能还是蛮有收益的一个尝试, 但并不是SA所说的那样 Another Giant Leap.



## **Rubin CPX芯片分析**

Rubin CPX是一颗基于 Rubin 架构但使用 GDDR7 的芯片。相对于基于 HBM 的平台，其不受 CoWoS 封装产能的限制，出货量可以很高。 芯片的基本规格如下：

![img](images/1758866431248-48d414a0-ee6e-4ede-bfad-b4de49f77eec.png)

- N3P工艺, 1 x Reticle Size
- NVFP4浮点算力<span style="color:magenta"> 30PFLOPS</span>
- 128GB GDDR7内存, 内存带宽2TB/s
- 互连仅支持PCIe Gen6x16, 支持单卡800Gbps带宽
- 内置了视频编解码, 但是是否带有光线追踪的RTCore未知
- 针对Attention计算中Softmax相关的指数计算SFU性能比B300提升了3倍

从芯片规格来看, 整体芯片还是对RTX 5090/6000 pro(GB202)的延续, 估计有<span style="color:magenta">192个SM</span>, 内存位宽为8 x 64bit-GDDR7, 如下图所示:

![图片](images/memory gddr7.png)



值得关注的是累计算力达到了 30PFLOPS, 相对于 Blackwell 这一代的 GB202(RTX 5090/6000pro), <span style="color:magenta">TensorCore 的算力提升了10倍</span>。 而30PFLOPS甚至超过了1x Reticle Size的Rubin Die(单颗Rubin在GTC25上宣布为50PFLOPS)。对于单个SM来看, TensorCore和SFU都将占用更大的芯片面积, 是否进一步砍掉一些高精度算力?



## **Rubin CPX NVL144**

### 标准VR NVL144

新一代的 Vera Rubin 架构还是延续 GB300 这样的 Oberon 机框结构,  <span style="color:magenta">单柜版本为 18 个 ComputeTray 和 9 个 SwitchTray</span>，单个 ComputeTray 包含了 4 颗Rubin GPU (每颗由2个Die封装)和 2 颗 Vera CPU 以及8颗 CX9：



![图片](images/rubin cpx nvl144.png)

- Vera CPU 包含 88 个 Arm Core。
- Rubin GPU 包含 288GB HBM4 显存，带宽为 13TB/s，50 PF FP4 算力，2 个 Die。
- Rubin GPU 的 <span style="color:magenta"> FP4 算力是 FP8 的 3x</span>>，与 B300 一致。
- Vera Fast Memory 为 (75TB - 288GB*72) / 36 = 1.5TB。
- NVLink6 带宽为 260/72 = 3.6 TB/s。
- ConnectX-9 网卡带宽为 28.8TB/s / 144 = 200 GB/s = 1600 Gbps(宣传值，实际值为800)。
-  <span style="color:magenta">HBM4 总带宽为 13*72=936TB/s</span>
  - 上图中1.4PB/s 实际是 HBM4 带宽 936 TB/s + Vera Fast Memory 带宽，因此后续统一称作 Memory 带宽，这种情况下 Vera CPU 的 Fast Memory 的带宽为 (1.4 PB/s - 936 TB/s)/36 = 0.46 TB/s / 36 = 13TB/s。
  - 后续会像 AMD MI400 Series 一样，采用 432GB 的 HBM4，显存带宽为 432/288*13 = 19.6 TB/s。





### CPX变体

ComputeTray 发布了变体：

1. 在标准ComputeTray上增加了 8 颗 Rubin CPX 构建的Vera Rubin CPX NVL144, 如下图所示。这种做法的优点是Rubin CPX直接PCIe总线连接到Vera上, KVCache传输的功耗较小，缺点是 <span style="color:red">Prefill 和 Decode 形成了固定的配比</span>，并不灵活.

   ![图片](images/cpxnvl144.png)

   

2. 另一种做法是采用两个机框的部署, 在VR CPX机柜中, 只有Vera CPU 和 Rubin CPX, 而没有Rubin芯片, 因此机柜只有18个ComputeTray, 没有NVLink SwitchTray. 而另一个机柜则是标准版的Vera Rubin NVL144机柜：

![图片](images/vr cpx dual rack.png)

这种做法相当于是分离式部署, 优点是可以根据自己的需求灵活的实现xPyD的配比, 但是缺点是 <span style="color:red">Vera CPU和CX9网卡的数量翻倍了, 而KVCache传输需要通过RDMA网络传输, 功耗和成本都更高</span>。



## 内部拓扑

标准的 Vera Rubin NVL144 机内 PCIe 拓扑如下左图所示, 而 Vera Rubin CPX NVL144 机内 PCIe 拓扑如下右图所示:



![图片](images/inside rubin pcie.png)



CX9 继续维持在 800Gbps, 主要是在 CX8 的基础上修正了一些bug和增加了某几个公司的一些功能需求. 因此<span style="color:red">CX9内置的PCIe Switch依旧约束在48 Lane</span>。

对于标准版的VR NVL144, 可以通过16x连接CPU, 16x连接Rubin, 并剩余16x可以连接NVMe盘, 但是考虑到前面板的空间约束, 单个CX9应该只能放置1块最多2块盘. 这个机型的好处是<span style="color:magenta">对于 ScaleOut 网络依旧可以通过 GPU-Direct-RDMA 进行通信</span>。

而对于 VR CPX NVL144 的版本, 需要留一根 PCIe Gen6x16 给 Rubin CPX, 因此判断在CX9+Rubin CPX的子卡上, 断开了PCIe的连接. CX9可以GDR到Rubin CPX, 利用ScaleOut网络执行Prefill的计算. 而<span style="color:red">CX9无法通过ScaleOut GDR连接到Rubin</span>.

而对于Vera Rubin CPX only的计算板拓扑如下所示:

![图片](images/cpx only.png)



其实对于其它的CSP更有可能选择Dual-Rack的方案, 但是并不需要官方的 CPX Only Rack 集成Vera CPU, 而是可以直接通过一个PCIe Switch Box旁至, 并使用PCIe AEC互连, 如下图所示:



![图片](images/dualrack.png)



这样的好处是既可以定制自己的网卡, 例如AWS Nitro, 又可以挂载更多的盘,并且还能维持Rubin GPU的 GDR 能力, 综合成本和功耗也应该小于官方的VR NVL144 + VR CPX only的Dual-Rack方案. 甚至还可以定制在PCIe Switch下挂载多个Rubin CPX芯片.

另一方面还可以像Meta那样, 对于VR NVL144依旧构建Dual-Rack的方案, 采用Vera CPU和Rubin 1:1配比. 然后左右两边并柜放置Prefill的PCIe Box.

## PD 分离

Nvidia官方有这样一个描述, Prefill 阶段是一个Compute Bound的计算. 对于 Coding/Agent 一类的LLM场景和视频生成(例如Veo3)/图片编辑(Nano Banana)场景来看, Prefill的长度通常会很长， 通用的GPU(例如Rubin)来看虽然算力/内存/NVLink互连带宽都兼顾了, 但是整体ROI来看并不好.



![图片](images/prefill.png)



因此构建一个Rubin CTX, 放弃对高内存带宽的需求, 使用GDDR7降低成本, 另一方面也放弃对高NVLink互连带宽的需求, <span style="color:magenta">专注于Prefill的场景</span>：



![图片](images/disaggregated infer.png)



我们来探讨几种情况下的Rubin CPX的P-D分离策略

### **Rubin CPX ScaleOut Prefill**

首先是针对所有的 Rubin CPX, 通过 ScaleOut RDMA 网络构成集群进行 Prefill 处理, 再通过PCIe将KVCache传输给Vera或者Rubin进行Decode Generation. 可以等效的看作每个ComputeTray 8张卡,每张卡800Gbps带宽构建的一个144卡的Prefill集群, 对于长Context而言, Attention计算应该没有太大的问题, 而ScaleOut带宽是否足够支撑后续的MoE 的EP并行?

其实在 Dual-Rack 的方案中就是这样的, Rubin CPX Only 的机柜只能通过 RDMA ScaleOut 网络通信. 同时又有大量的 KVCache 也要通过 RDMA 网络传输到 VR NVL144机柜. KVCache传输和EP的干扰也是一个麻烦事, 毕竟和单机柜的VR CPX NVL144相比, <span style="color:red">RDMA 传输 KVCache 比起机内直接 D2H copy 带来了一些不确定性</span>

### **Rubin CPX with NVLink**

是否可以借助NVLink的大带宽优势呢? 也就是说 Attention 计算完了以后, 传递一份 Token 给Vera, 然后通过NVLink dispatch多份到其它ComputeTray, 然后还可以借助DeepSeek-V3这样的Group方案来做2级的分发, 即按照一个 ComputeTray 一个 Group 的方式分配 Experts, 然后在 Nvlink 上减少 Dispatch 的份数, 避免NVLink 上对其它 Decoding 的任务产生影响(例如Rubin的L2Cache污染/HBM带宽占用等).  <span style="color:magenta">这是一个可以探索的方向</span>.

这个方案可以降低一些通信量, 并且把 NVLink 的一些带宽用起来, 但是 Vera CPU 并不一定能在 PCIe 上扛住这么大的带宽, 毕竟8x800Gbps已经达到800GB/s了, 这么大的流量穿越 Vera 还是有一些潜在的问题的. 那么前面提到的基于PCIe Switch, 让Rubin CPX直接PCIe P2P拷贝到Rubin可能是比官方的VR CPX NVL144更好的方案?

### Rubin CPX Attention with Rubin FFN

是否能够借助 Rubin 来做一些 Expert 的计算? 但是需要考虑整个 Timeline 如何去做 Overlap，并且不影响 Decode. 当然 Decode 阶段 NVLink 带宽和 GEMM 本身的效率来看也需要攒够更大的 Batch 提升 MFU.

可能这种方案有收益, 但额外的极长的Context在Rubin上计算可能对Decode也带来了负收益, 这些取决于SLA标准如何定义, 然后平台如何取舍.

### 混合调度方案

是否还是在NVL144中配置xPyD的方案, 仅对SeqLen很长的任务Offload到Rubin CPX处理? 这也是一个潜在的可以尝试的调度策略. 因为我们还需要考虑KVcache对显存的占用. Rubin CPX毕竟只有128GB的显存. 例如对于一个256K Seqlen的Prefill最高能到多少并发也需要根据模型计算的.



NVL576


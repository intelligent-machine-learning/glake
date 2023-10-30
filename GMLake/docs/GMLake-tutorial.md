# GMLake-tutorial
## GMLake环境变量
### vmmDefragment
该环境变量代表了是否开启GMLake的碎片stitch功能，即是否会把碎片拼接起来。可接受的值，1代表开启，0代表关闭，默认值为1。值为0时代表处理方式按照原始PyTorch中的cache机制处理

### fragment_limit
该环境变量代表了针对size大于或等于fragment_limit值的block做碎片stitch。可接受的值为正整数，单位为Byte，默认值是512MB

### reuseLimit
该环境变量代表了在复用stitch block时的上限，即请求的block size * reuseLimit代表可复用的stitch block的size上限，超过上限，该stitch block不可被复用。可接受的值为正浮点数，默认值是10

### defragment_level
该环境变量代表了做stitch操作的时机，默认值为0，只要空闲的block足够拼接一个请求的block，就会做stitch操作。当值为1时，代表只有当显存不够时才开始做stitch操作

### auto_gc_limits
该环境变量代表了做GC操作的阈值，单位是GB，默认值是1000，代表了当单个GPU上有超过1000GB的stitch block时会做GC，该GC操作仅代表GMLake中针对stitch block的GC操作，不会真正释放原始的物理block

## 环境变量设置说明
上述环境变量在跑不同的模型时需要做相应地调整，尤其是针对fragment_limit的值，由于nvidia一些接口调用次数存在限制，当单个机器上使用的GPU卡数越多时，每个GPU卡上stitch操作做太多之后需要做GC操作，此时可观察，如果打印日志中fuse的日志很多时，并且运行很长一段时间后仍然无法稳定下来，可适当增大fragment_limit的值，减少每个GPU卡上的stitch操作。而如果显存碎片效果减少不理想时，也可以适当减少fragment_limit的值，此时可有效增加碎片的优化效果。

## 模型测试
我们已经针对多个模型在不同的场景下做了很多的测试，详细情况请参考我们之前的文章。本次使用文档我们以opt-1.3b的模型finetune训练来作为benchmark。选用的训练代码为DeepSpeedExamples中step1 finetune的代码，为了方便作为example测试，我们将这部分代码单独剥离出来后放入到我们做好的一个镜像中，并且将测试集以及模型权重都放到了镜像中，可以直接下载使用，复现我们GMLake的效果。环境为8卡A100的机器。

按照如下命令拉取并启动docker镜像
```
sudo docker run -td --net=host --ipc=host -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all easydl/glake:v1
```
进入镜像中
```
sudo docker exec -it container-id bash
```
我们再镜像中的root目录下放置了两个PyTorch libc10_cuda.so的库，分别对应的是原始PyTorch的库，以及使用了GMLake的库，用户可以选择不同的库进行测试，也可以直接用GMLake的库，用vmmDefragment环境变量来关闭和开启GMLake的碎片拼接功能。

首先，我们测试原始的PyTorch版本，先按照如下命令使用原始PyTorch的库
```
cp /root/pytorch2-origin-lib/libc10_cuda.so /opt/conda/lib/python3.8/site-packages/torch/lib
```

进入到模型的测试代码中
```
cd /GMLake-test/DeepSpeed-Chat/training/step1_supervised_finetuning/
```
修改模型的benchmark命令，在benchmark中，可以设置使用的GPU数量、模型的batch_size、使用的显存优化策略等，我们按照如下配置来进行测试，使用4个GPU，模型的batch_size选择160，使用的显存优化策略为开启LoRA和recomputing，分布式策略使用DeepSpeed zero3，即按照如下配置benchmark的脚本
```
vim training_scripts/single_node/benchmark.sh
for BS in 160
do
for GPU_NUM in 4
do
        bash training_scripts/single_node/finetune.sh $GPU_NUM $BS facebook/opt-1.3b False 1 1 0
done
done
```
然后运行如下的命令
```
bash training_scripts/single_node/benchmark.sh
```
此时，可以查看当前模型的训练日志
```
tail -f output/output_opt-1.3b/training_opt-1.3b_G4_B160_GLAKE-False_L1_R1_O0_GC1000_FL268435456_RL10ori.log
```
当运行一段时间之后，可以查看该模型打印的torch.cuda.memory_summary的信息，可以重点关注Peak Usage这一列中Active memory和GPU reserved memory两个值之间的比值，该值代表本次训练显存的有效利用率，用原始PyTorch测试，显存利用率为64%左右

然后来测试相同配置情况下GMLake的效果，首先执行如下命令来替换为GMLake的库
```
cp /root/GMLake-lib/libc10_cuda.so /opt/conda/lib/python3.8/site-packages/torch/lib
```

进入到模型的测试代码中
```
cd /GMLake-test/DeepSpeed-Chat/training/step1_supervised_finetuning/
```

按照如下配置来修改benchmark的脚本
```
vim training_scripts/single_node/benchmark.sh
for BS in 160
do
for GPU_NUM in 4
do
        bash training_scripts/single_node/finetune.sh $GPU_NUM $BS facebook/opt-1.3b True 1 1 0
done
done
```
可以查看下当前GMLake针对一些环境变量的设置，由于opt-1.3b的模型比较小，所以将fragLimit的值设置为了256MB
```
export vmmDefragment=1
export autoGC=10000
export fragLimit=268435456
export reuseLimit=10
export defragLevel=0
```
然后运行如下的命令
```
bash training_scripts/single_node/benchmark.sh
```

此时，可以查看当前模型的训练日志
```
tail -f output/output_opt-1.3b/training_opt-1.3b_G4_B160_GLAKE-True_L1_R1_O0_GC1000_FL268435456_RL10ori.log
```
当运行一段时间后，查看当前模型打印的torch.cuda.memory_summary，Peak Usage这一列中Active memory和GPU reserved memory两个值之间的比值，用GMLake测试，显存利用率为88%，相比原始PyTorch，我们将显存利用率提升了24%，占用的显存减少了18G

以上仅是针对一种场景下的测试，选用的显存优化策略为LoRA和recomputing，使用DeepSpeed zero3做分布式策略，也可以按照我们文中测试的，配置不同的显存优化策略，或者调整不同的batch_size进行测试

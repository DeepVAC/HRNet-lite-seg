# HRNet-lite-seg
DeepVAC-compliant HRNet-lite implementation for segmentation.

# 简介
本项目实现了符合DeepVAC规范的HRNet-lite-seg 。

### 项目依赖

- deepvac >= 0.5.7
- pytorch >= 1.8.0
- torchvision >= 0.7.0
- opencv-python
- numpy

# 如何运行本项目

## 1. 阅读[DeepVAC规范](https://github.com/DeepVAC/deepvac)
可以粗略阅读，建立起第一印象。

## 2. 准备运行环境
使用Deepvac规范指定[Docker镜像](https://github.com/DeepVAC/deepvac#2-%E7%8E%AF%E5%A2%83%E5%87%86%E5%A4%87)。

## 3. 准备数据集

- TODO

- 在config.py中修改如下配置：
```python
config.train_txt = './data/train.txt'
config.val_txt = './data/val.txt'
config.sample_path_prefix = 'your train images dir'
```

## 4. 训练相关配置

- dataloader相关配置

```python
config.datasets.FileLineCvSegWithMetaInfoDataset = AttrDict()
config.datasets.FileLineCvSegWithMetaInfoDataset.cached_data_file = 'data/clothes.p'
config.datasets.FileLineCvSegWithMetaInfoDataset.classes = config.cls_num
config.datasets.FileLineCvSegWithMetaInfoDataset.norm_val = 1.10
config.data = FileLineCvSegWithMetaInfoDataset(config, config.train_txt, config.sample_path_prefix)()
config.datasets.FileLineCvSegDataset = AttrDict()
config.datasets.FileLineCvSegDataset.composer = LiteHRNetTrainComposer(config)


config.batch_size = 8
config.num_workers = 3
config.core.LiteHRNetTrain.train_dataset = FileLineCvSegDataset(config, config.train_txt, config.delimiter, config.sample_path_prefix)
config.core.LiteHRNetTrain.train_loader = torch.utils.data.DataLoader(config.core.LiteHRNetTrain.train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=config.pin_memory)

```
## 5. 训练

### 5.1 单卡训练
执行命令：

```bash
python3 train.py
```

### 5.2 分布式训练

在config.py中修改如下配置：
```python
#dist_url，单机多卡无需改动，多机训练一定要修改
config.core.LiteHRNetTrain.dist_url = "tcp://localhost:27030"

#rank的数量，一定要修改
config.core.LiteHRNetTrain.world_size = 2
```
然后执行命令：

```bash
python train.py --rank 0 --gpu 0
python train.py --rank 1 --gpu 1
```


## 6. 测试

- 测试相关配置

```python
config.core.LiteHRNetTest = config.core.LiteHRNetTrain.clone()
config.core.LiteHRNetTest.test_sample_path = 'your test image dir'
config.core.LiteHRNetTest.model_path = 'your trained model path'
```

- 加载模型(*.pth)

```python
config.core.LiteHRNetTest.model_path = <trained-model-path>
```

- 运行测试脚本：

```bash
python3 test.py
```
## 7. 使用trace模型/script模型
如果训练过程中开启config.cast.TraceCast（或者config.cast.ScriptCast)开关，可以在测试过程中转化torchscript模型     

- 转换torchscript模型(*.pt)     

```python
# trace
config.cast.TraceCast = AttrDict()
config.cast.TraceCast.model_dir = "./trace.pt"

# script
config.cast.ScriptCast = AttrDict()
config.cast.ScriptCast.model_dir = "./script.pt"
```

按照步骤6完成测试，torchscript模型将保存至model_dir指定文件位置      

- 加载torchscript模型

```python
config.core.LiteHRNetTrain.jit_model_path = <torchscript-model-path>
config.core.LiteHRNetTest.jit_model_path = <torchscript-model-path>
```

## 8. 使用静态量化模型
如果训练过程中未开启config.cast.TraceCast开关，可以在测试过程中转化静态量化模型     
- 转换静态模型(*.sq)     

```python
# trace
config.cast.TraceCast.static_quantize_dir = "./trace.sq"

# script
config.cast.ScriptCast.static_quantize_dir = "./script.sq"
```
按照步骤6完成测试，静态量化模型将保存至config.static_quantize_dir指定文件位置      

- 加载静态量化模型

```python
config.core.LiteHRNetTrain.jit_model_path = <static-quantize-model-path>
config.core.LiteHRNetTest.jit_model_path = <static-quantize-model-path>
```
- 动态量化模型对应的配置参数为config.cast.TraceCast.dynamic_quantize_dir(或者config.cast.ScriptCast.dynamic_quantize_dir)

## 9. 更多功能
如果要在本项目中开启如下功能：
- 预训练模型加载
- checkpoint加载
- 使用tensorboard
- 启用TorchScript
- 转换ONNX
- 转换NCNN
- 转换CoreML
- 开启量化
- 开启自动混合精度训练

请参考[DeepVAC](https://github.com/DeepVAC/deepvac)
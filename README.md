# Grad-CAM.pytorch

​          pytorch 实现[Grad-CAM:Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/pdf/1610.02391) 和

[Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks](https://arxiv.org/pdf/1710.11063.pdf)

1. [依赖](#依赖)
2. [使用方法](#使用方法)
3. [样例分析](#样例分析)<br>
   3.1 [单个对象](#单个对象)<br>
   3.3 [多个对象](#多个对象)<br>
4. [总结](#总结)

**Grad-CAM整体架构**

![](examples/grad-cam.jpg)



**Grad-CAM++与Grad-CAM的异同**

![](examples/Grad-CAM++.png)



## 依赖

```wiki
python 3.6.x
pytoch 1.0.1+
torchvision 0.2.2
opencv-python
matplotlib
scikit-image
numpy
```



## 使用方法

```shell
python main.py --image-path examples/pic1.jpg \
               --network densenet121 \
               --weight-path /opt/pretrained_model/densenet121-a639ec97.pth
```

**参数说明**：

- image-path：需要可视化的图像路径(可选,默认`./examples/pic1.jpg`)

- network: 网络名称(可选,默认`resnet50`)
- weight-path: 网络对应的与训练参数权重路径(可选,默认从pytorch官网下载对应的预训练权重)
- layer-name: Grad-CAM使用的层名(可选,默认最后一个卷积层)
- class-id：Grad-CAM和Guided Back Propagation反向传播使用的类别id（可选,默认网络预测的类别)
- output-dir：可视化结果图像保存目录(可选，默认`results`目录)



## 样例分析

### 单个对象

**原始图像**

![](/examples/pic1.jpg)

**效果**

| network      | HeatMap                                   | Grad-CAM                              | HeatMap++                                   | Grad-CAM++                              | Guided backpropagation               | Guided Grad-CAM                          |
| ------------ | ----------------------------------------- | ------------------------------------- | ------------------------------------------- | --------------------------------------- | ------------------------------------ | ---------------------------------------- |
| vgg16        | ![](results/pic1-vgg16-heatmap.jpg)       | ![](results/pic1-vgg16-cam.jpg)       | ![](results/pic1-vgg16-heatmap++.jpg)       | ![](results/pic1-vgg16-cam++.jpg)       | ![](results/pic1-vgg16-gb.jpg)       | ![](results/pic1-vgg16-cam_gb.jpg)       |
| vgg19        | ![](results/pic1-vgg19-heatmap.jpg)       | ![](results/pic1-vgg19-cam.jpg)       | ![](results/pic1-vgg19-heatmap++.jpg)       | ![](results/pic1-vgg19-cam++.jpg)       | ![](results/pic1-vgg19-gb.jpg)       | ![](results/pic1-vgg19-cam_gb.jpg)       |
| resnet50     | ![](results/pic1-resnet50-heatmap.jpg)    | ![](results/pic1-resnet50-cam.jpg)    | ![](results/pic1-resnet50-heatmap++.jpg)    | ![](results/pic1-resnet50-cam++.jpg)    | ![](results/pic1-resnet50-gb.jpg)    | ![](results/pic1-resnet50-cam_gb.jpg)    |
| resnet101    | ![](results/pic1-resnet101-heatmap.jpg)   | ![](results/pic1-resnet101-cam.jpg)   | ![](results/pic1-resnet101-heatmap++.jpg)   | ![](results/pic1-resnet50-cam++.jpg)    | ![](results/pic1-resnet101-gb.jpg)   | ![](results/pic1-resnet101-cam_gb.jpg)   |
| densenet121  | ![](results/pic1-densenet121-heatmap.jpg) | ![](results/pic1-densenet121-cam.jpg) | ![](results/pic1-densenet121-heatmap++.jpg) | ![](results/pic1-densenet121-cam++.jpg) | ![](results/pic1-densenet121-gb.jpg) | ![](results/pic1-densenet121-cam_gb.jpg) |
| inception_v3 | ![](results/pic1-inception-heatmap.jpg)   | ![](results/pic1-inception-cam.jpg)   | ![](results/pic1-inception-heatmap++.jpg)   | ![](results/pic1-inception-cam++.jpg)   | ![](results/pic1-inception-gb.jpg)   | ![](results/pic1-inception-cam_gb.jpg)   |
|              |                                           |                                       |                                             |                                         |                                      |                                          |

### 多个对象

​         对应多个图像Grad-CAM++比Grad-CAM覆盖要更全面一些，这也是Grad-CAM++最主要的优势

**原始图像**

![](/examples/multiple_dogs.jpg)

**效果**

| network      | HeatMap                                   | Grad-CAM                              | HeatMap++                                   | Grad-CAM++                              | Guided backpropagation               | Guided Grad-CAM                          |
| ------------ | ----------------------------------------- | ------------------------------------- | ------------------------------------------- | --------------------------------------- | ------------------------------------ | ---------------------------------------- |
| vgg16        | ![](results/multiple_dogs-vgg16-heatmap.jpg)       | ![](results/multiple_dogs-vgg16-cam.jpg)       | ![](results/multiple_dogs-vgg16-heatmap++.jpg)       | ![](results/multiple_dogs-vgg16-cam++.jpg)       | ![](results/multiple_dogs-vgg16-gb.jpg)       | ![](results/multiple_dogs-vgg16-cam_gb.jpg)       |
| vgg19        | ![](results/multiple_dogs-vgg19-heatmap.jpg)       | ![](results/multiple_dogs-vgg19-cam.jpg)       | ![](results/multiple_dogs-vgg19-heatmap++.jpg)       | ![](results/multiple_dogs-vgg19-cam++.jpg)       | ![](results/multiple_dogs-vgg19-gb.jpg)       | ![](results/multiple_dogs-vgg19-cam_gb.jpg)       |
| resnet50     | ![](results/multiple_dogs-resnet50-heatmap.jpg)    | ![](results/multiple_dogs-resnet50-cam.jpg)    | ![](results/multiple_dogs-resnet50-heatmap++.jpg)    | ![](results/multiple_dogs-resnet50-cam++.jpg)    | ![](results/multiple_dogs-resnet50-gb.jpg)    | ![](results/multiple_dogs-resnet50-cam_gb.jpg)    |
| resnet101    | ![](results/multiple_dogs-resnet101-heatmap.jpg)   | ![](results/multiple_dogs-resnet101-cam.jpg)   | ![](results/multiple_dogs-resnet101-heatmap++.jpg)   | ![](results/multiple_dogs-resnet50-cam++.jpg)    | ![](results/multiple_dogs-resnet101-gb.jpg)   | ![](results/multiple_dogs-resnet101-cam_gb.jpg)   |
| densenet121  | ![](results/multiple_dogs-densenet121-heatmap.jpg) | ![](results/multiple_dogs-densenet121-cam.jpg) | ![](results/multiple_dogs-densenet121-heatmap++.jpg) | ![](results/multiple_dogs-densenet121-cam++.jpg) | ![](results/multiple_dogs-densenet121-gb.jpg) | ![](results/multiple_dogs-densenet121-cam_gb.jpg) |
| inception_v3 | ![](results/multiple_dogs-inception-heatmap.jpg)   | ![](results/multiple_dogs-inception-cam.jpg)   | ![](results/multiple_dogs-inception-heatmap++.jpg)   | ![](results/multiple_dogs-inception-cam++.jpg)   | ![](results/multiple_dogs-inception-gb.jpg)   | ![](results/multiple_dogs-inception-cam_gb.jpg)   |
|              |                                           |                                       |                                             |                                         |                                      |                                          |

 

## 总结

- vgg模型的Grad-CAM并没有覆盖整个对象,相对来说resnet和denset覆盖更全,特别是densenet;从侧面说明就模型的泛化和鲁棒性而言densenet>resnet>vgg
- Grad-CAM++相对于Grad-CAM也是覆盖对象更全面，特别是对于同一个类别有多个实例的情况下,Grad-CAM可能只覆盖部分对象，Grad-CAM++基本覆盖所有对象;但是这仅仅对于vgg而言,想densenet直接使用Grad-CAM也基本能够覆盖所有对象
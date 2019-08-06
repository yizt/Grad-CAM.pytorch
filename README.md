# Grad-CAM.pytorch
​          pytorch 实现[Grad-CAM:Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/pdf/1610.02391)

![](D:\pyspace\Grad-CAM.pytorch\examples\grad-cam.jpg)



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



## 样例

![](D:\pyspace\Grad-CAM.pytorch\examples\pic1.jpg)

## 结果

| network      | HeatMap                                   | Grad-CAM                              | Guided backpropagation               | Guided Grad-CAM                          |
| ------------ | ----------------------------------------- | ------------------------------------- | ------------------------------------ | ---------------------------------------- |
| vgg16        | ![](results/pic1-vgg16-heatmap.jpg)       | ![](results/pic1-vgg16-cam.jpg)       | ![](results/pic1-vgg16-gb.jpg)       | ![](results/pic1-vgg16-cam_gb.jpg)       |
| vgg19        | ![](results/pic1-vgg19-heatmap.jpg)       | ![](results/pic1-vgg19-cam.jpg)       | ![](results/pic1-vgg19-gb.jpg)       | ![](results/pic1-vgg19-cam_gb.jpg)       |
| resnet50     | ![](results/pic1-resnet50-heatmap.jpg)    | ![](results/pic1-resnet50-cam.jpg)    | ![](results/pic1-resnet50-gb.jpg)    | ![](results/pic1-resnet50-cam_gb.jpg)    |
| resnet101    | ![](results/pic1-resnet101-heatmap.jpg)   | ![](results/pic1-resnet101-cam.jpg)   | ![](results/pic1-resnet101-gb.jpg)   | ![](results/pic1-resnet101-cam_gb.jpg)   |
| densenet121  | ![](results/pic1-densenet121-heatmap.jpg) | ![](results/pic1-densenet121-cam.jpg) | ![](results/pic1-densenet121-gb.jpg) | ![](results/pic1-densenet121-cam_gb.jpg) |
| inception_v3 | ![](results/pic1-inception-heatmap.jpg)   | ![](results/pic1-inception-cam.jpg)   | ![](results/pic1-inception-gb.jpg)   | ![](results/pic1-inception-cam_gb.jpg)   |
|              |                                           |                                       |                                      |                                          |


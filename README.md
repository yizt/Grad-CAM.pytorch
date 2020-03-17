# Grad-CAM.pytorch

​          pytorch 实现[Grad-CAM:Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/pdf/1610.02391) 和

[Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks](https://arxiv.org/pdf/1710.11063.pdf)

1. [依赖](#依赖)

2. [使用方法](#使用方法)

3. [样例分析](#样例分析)<br>
   3.1 [单个对象](#单个对象)<br>
   3.3 [多个对象](#多个对象)<br>

4. [总结](#总结)

5. [目标检测](#目标检测)<br>
   5.1 [detectron2安装](#detectron2安装)<br>
   5.2 [测试](#测试)<br>
   5.3 [Grad-CAM结果](#Grad-CAM结果)<br>
   5.4 [总结](#总结)

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
| mobilenet_v2 | ![](results/pic1-mobilenet_v2-heatmap.jpg)   | ![](results/pic1-mobilenet_v2-cam.jpg)   | ![](results/pic1-mobilenet_v2-heatmap++.jpg)   | ![](results/pic1-mobilenet_v2-cam++.jpg)   | ![](results/pic1-mobilenet_v2-gb.jpg)   | ![](results/pic1-mobilenet_v2-cam_gb.jpg)   |
| shufflenet_v2 | ![](results/pic1-shufflenet_v2-heatmap.jpg)   | ![](results/pic1-shufflenet_v2-cam.jpg)   | ![](results/pic1-shufflenet_v2-heatmap++.jpg)   | ![](results/pic1-shufflenet_v2-cam++.jpg)   | ![](results/pic1-shufflenet_v2-gb.jpg)   | ![](results/pic1-shufflenet_v2-cam_gb.jpg)   |

### 多个对象

​         对应多个图像Grad-CAM++比Grad-CAM覆盖要更全面一些，这也是Grad-CAM++最主要的优势

**原始图像**

![](./examples/multiple_dogs.jpg)

**效果**

| network      | HeatMap                                   | Grad-CAM                              | HeatMap++                                   | Grad-CAM++                              | Guided backpropagation               | Guided Grad-CAM                          |
| ------------ | ----------------------------------------- | ------------------------------------- | ------------------------------------------- | --------------------------------------- | ------------------------------------ | ---------------------------------------- |
| vgg16        | ![](results/multiple_dogs-vgg16-heatmap.jpg)       | ![](results/multiple_dogs-vgg16-cam.jpg)       | ![](results/multiple_dogs-vgg16-heatmap++.jpg)       | ![](results/multiple_dogs-vgg16-cam++.jpg)       | ![](results/multiple_dogs-vgg16-gb.jpg)       | ![](results/multiple_dogs-vgg16-cam_gb.jpg)       |
| vgg19        | ![](results/multiple_dogs-vgg19-heatmap.jpg)       | ![](results/multiple_dogs-vgg19-cam.jpg)       | ![](results/multiple_dogs-vgg19-heatmap++.jpg)       | ![](results/multiple_dogs-vgg19-cam++.jpg)       | ![](results/multiple_dogs-vgg19-gb.jpg)       | ![](results/multiple_dogs-vgg19-cam_gb.jpg)       |
| resnet50     | ![](results/multiple_dogs-resnet50-heatmap.jpg)    | ![](results/multiple_dogs-resnet50-cam.jpg)    | ![](results/multiple_dogs-resnet50-heatmap++.jpg)    | ![](results/multiple_dogs-resnet50-cam++.jpg)    | ![](results/multiple_dogs-resnet50-gb.jpg)    | ![](results/multiple_dogs-resnet50-cam_gb.jpg)    |
| resnet101    | ![](results/multiple_dogs-resnet101-heatmap.jpg)   | ![](results/multiple_dogs-resnet101-cam.jpg)   | ![](results/multiple_dogs-resnet101-heatmap++.jpg)   | ![](results/multiple_dogs-resnet50-cam++.jpg)    | ![](results/multiple_dogs-resnet101-gb.jpg)   | ![](results/multiple_dogs-resnet101-cam_gb.jpg)   |
| densenet121  | ![](results/multiple_dogs-densenet121-heatmap.jpg) | ![](results/multiple_dogs-densenet121-cam.jpg) | ![](results/multiple_dogs-densenet121-heatmap++.jpg) | ![](results/multiple_dogs-densenet121-cam++.jpg) | ![](results/multiple_dogs-densenet121-gb.jpg) | ![](results/multiple_dogs-densenet121-cam_gb.jpg) |
| inception_v3 | ![](results/multiple_dogs-inception-heatmap.jpg)   | ![](results/multiple_dogs-inception-cam.jpg)   | ![](results/multiple_dogs-inception-heatmap++.jpg)   | ![](results/multiple_dogs-inception-cam++.jpg)   | ![](results/multiple_dogs-inception-gb.jpg)   | ![](results/multiple_dogs-inception-cam_gb.jpg)   |
| mobilenet_v2 | ![](results/multiple_dogs-mobilenet_v2-heatmap.jpg)   | ![](results/multiple_dogs-mobilenet_v2-cam.jpg)   | ![](results/multiple_dogs-mobilenet_v2-heatmap++.jpg)   | ![](results/multiple_dogs-mobilenet_v2-cam++.jpg)   | ![](results/multiple_dogs-mobilenet_v2-gb.jpg)   | ![](results/multiple_dogs-mobilenet_v2-cam_gb.jpg)   |
| shufflenet_v2 | ![](results/multiple_dogs-shufflenet_v2-heatmap.jpg)   | ![](results/multiple_dogs-shufflenet_v2-cam.jpg)   | ![](results/multiple_dogs-shufflenet_v2-heatmap++.jpg)   | ![](results/multiple_dogs-shufflenet_v2-cam++.jpg)   | ![](results/multiple_dogs-shufflenet_v2-gb.jpg)   | ![](results/multiple_dogs-shufflenet_v2-cam_gb.jpg)   |

 

## 总结

- vgg模型的Grad-CAM并没有覆盖整个对象,相对来说resnet和denset覆盖更全,特别是densenet;从侧面说明就模型的泛化和鲁棒性而言densenet>resnet>vgg
- Grad-CAM++相对于Grad-CAM也是覆盖对象更全面，特别是对于同一个类别有多个实例的情况下,Grad-CAM可能只覆盖部分对象，Grad-CAM++基本覆盖所有对象;但是这仅仅对于vgg而言,想densenet直接使用Grad-CAM也基本能够覆盖所有对象
- MobileNet V2的Grad-CAM覆盖也很全面
- Inception V3和MobileNet V2的Guided backpropagation图轮廓很模糊，但是ShuffleNet V2的轮廓则比较清晰



## 目标检测

​        有位网友@SHAOSIHAN问道怎样在目标检测中使用Grad-CAM;在Grad-CAM和Grad-CAM++论文中都没有提及对目标检测生成CAM图。我想主要有两个原因：

a) 目标检测不同于分类，分类网络只有一个分类损失，而且所有网络都是一样的(几个类别最后一层就是几个神经元)，最后的预测输出都是单一的类别得分分布。目标检测则不同，输出都不是单一的，而且不同的网络如Faster R-CNN, CornerNet,CenterNet,FCOS，它们的建模方式不一样，输出的含义都不相同。所以不会有统一的生成Grad-CAM图的方法。

b) 分类属于弱监督，通过CAM可以了解网络预测时主要关注的空间位置，也就是"看哪里"，对分析问题有实际的价值；而目标检测，本身是强监督，预测边框就直接指示了“看哪里”。

​         

​        这里以detetron2中的faster-rcnn网络为例，生成Grad-CAM图。主要思路是直接获取预测分值最高的边框;将该边框的预测分值反向传播梯度到，该边框对应的proposal 边框的feature map上，生成此feature map的CAM图。



### detectron2安装

a) 下载

```shell
git clone https://github.com/facebookresearch/detectron2.git
```



b) 修改`detectron2/modeling/roi_heads/fast_rcnn.py`文件中的`fast_rcnn_inference_single_image`函数，主要是增加索引号，记录分值高的预测边框是由第几个proposal边框生成的；修改后的`fast_rcnn_inference_single_image`函数如下：

```python
def fast_rcnn_inference_single_image(
        boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    indices = torch.arange(start=0, end=scores.shape[0], dtype=int)
    indices = indices.expand((scores.shape[1], scores.shape[0])).T
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        indices = indices[valid_mask]
    scores = scores[:, :-1]
    indices = indices[:, :-1]

    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # Filter results based on detection scores
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]

    scores = scores[filter_mask]
    indices = indices[filter_mask]
    # Apply per-class NMS
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
    indices = indices[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    result.indices = indices
    return result, filter_inds[:, 0]
```



c) 安装;如遇到问题，请参考[detectron2](https://github.com/facebookresearch/detectron2)；不同操作系统安装有差异

```shell
cd detectron2
pip install -e .
```



### 测试

a) 预训练模型下载

```shell
wget https://dl.fbaipublicfiles.com/detectron2/PascalVOC-Detection/faster_rcnn_R_50_C4/142202221/model_final_b1acc2.pkl
```



b) 测试Grad-CAM图像生成

​          在本工程目录下执行如下命令

```shell
export KMP_DUPLICATE_LIB_OK=TRUE
python detection/demo.py --config-file detection/faster_rcnn_R_50_C4.yaml \
--input ./examples/pic1.jpg \
--opts MODEL.WEIGHTS /Users/yizuotian/pretrained_model/model_final_b1acc2.pkl MODEL.DEVICE cpu
```



### Grad-CAM结果

| 原始图像                 | 检测边框                                  | Grad-CAM HeatMap                      | Grad-CAM++ HeatMap                      | 边框预测类别 |
| ------------------------ | ----------------------------------------- | ------------------------------------- | --------------------------------------- | ------------ |
| ![](./examples/pic1.jpg) | ![](./results/pic1-frcnn-predict_box.jpg) | ![](./results/pic1-frcnn-heatmap.jpg) | ![](./results/pic1-frcnn-heatmap++.jpg) | Dog          |
| ![](./examples/pic2.jpg) | ![](./results/pic2-frcnn-predict_box.jpg) | ![](./results/pic2-frcnn-heatmap.jpg) | ![](./results/pic2-frcnn-heatmap++.jpg) | Aeroplane    |
| ![](./examples/pic3.jpg) | ![](./results/pic3-frcnn-predict_box.jpg) | ![](./results/pic3-frcnn-heatmap.jpg) | ![](./results/pic3-frcnn-heatmap++.jpg) | Person       |
| ![](./examples/pic4.jpg) | ![](./results/pic4-frcnn-predict_box.jpg) | ![](./results/pic4-frcnn-heatmap.jpg) | ![](./results/pic4-frcnn-heatmap++.jpg) | Horse        |


### 总结

​          对于目标检测Grad-CAM++的效果并没有比Grad-CAM效果好，推测目标检测中预测边框已经是单个对象了,Grad-CAM++在多个对象的情况下优于Grad-CAM
# Post-processing of the drivable area(C++)![GitHub](https://img.shields.io/github/license/daohu527/Dig-into-Apollo.svg?style=popout)

Modify the verified driveable area post-processing and modify it to the C++ version.  
Workflow:![workflow](https://github.com/zhangbanxian123/Driving-area-detection/blob/master/workflow.png)



## input and output

**input**:The label map output by the semantic segmentation network.  
**output**:colored label map of lanes.  
**example img of input and output**:
![input](https://github.com/zhangbanxian123/Driving-area-detection/blob/master/gray.png)  
![output](https://github.com/zhangbanxian123/Driving-area-detection/blob/master/trt_img.jpg)  

Because the output label of the semantic segmentation network is [0,1,2], which is close to 0, so the input of this work viewed as black.

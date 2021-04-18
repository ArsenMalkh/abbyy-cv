# Results
## Keypoint matching
Использоался матчинг кейпоинтов по их дексрипторам, для shi-thomasi использовался сифт
### Repeatability:
| method |  repeatability |  
| ------ | ------ | 
| shi-thomasi | 0.67 |
| sift | 0.41 |
| orb | 0.5 |
### Percent of repeated points on Nth img
Доля повторенных точек с первого изображения на N-ом изображении
![N rep](https://github.com/armored-guitar/abbyy-cv/blob/master/keypoint_repeatability/output/in_nth_images_matching.jpg?raw=true)
### Percent of repeated points in N and more images
Доля точек, повторенных минимум на N изображениях
![N rep](https://github.com/armored-guitar/abbyy-cv/blob/master/keypoint_repeatability/output/in_n_images_matching.jpg?raw=true)

## Motion Estimation
Использовался матчинг на основе оценки движения (phase correlation)
### Repeatability:
| method |  repeatability |  
| ------ | ------ | 
| shi-thomasi | 0.49 |
| sift | 0.47 |
| orb | 0.48 |
### Percent of repeated points on Nth img
Доля повторенных точек с первого изображения на N-ом изображении
![N rep](https://github.com/armored-guitar/abbyy-cv/blob/master/keypoint_repeatability/output/in_nth_images_motion_est.jpg?raw=true)
### Percent of repeated points in N and more images
Доля точек, повторенных минимум на N изображениях
![N rep](https://github.com/armored-guitar/abbyy-cv/blob/master/keypoint_repeatability/output/in_n_images_motion_est.jpg?raw=true)
## Time for one keypoint
| method |  time for one keypoint, us |  
| ------ | ------ | 
| shi-thomasi | 2.32 |
| sift | 22.37 |
| orb | 2.85 |





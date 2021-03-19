# Find image translation
## Использование
```
python3 main.py -h
  --imgs_path IMGS_PATH input images path
  --nms NMS             nms size
  --conf CONF           confidence for keypoint detection
  --nn_tresh NN_TRESH   threshold for nearest neighbors algorithm
  --save_path SAVE_PATH path to save results csv
```
## Результат:
Время на одну картинку: 0.054с, время на все картинки 0.644c
### Сравнение с phase correlation
первые две колонки - с использованием SuperPoint, вторые две - phase correlation

| x superpoint |  y superpoint | x phase correlation | y phase correlation |
| ------ | ------ | ------ | ------ |
|5.4 | 0.1 | 5.3 | -0.5|
|0.7 | 0.8 | 1.7 | 0.3|
|3.3 | -3.4 | 2.3 | -2.9|
|1.2 | -5.5 | 0.6 | -5.1|
|1.3 | -7.1 | 6.1 | -7.7|
|-1.9 | -4.9 | -2.7 | -5.4|
|-6.3 | -6.9 | -1.8 | -7.9|
|-2.1 | -13.3 | 1.2 | -14.6|
|-0.6 | -21.2 | -0.6 | -19.1|
|1.2 | -24.0 | 2.1 | -22.9|
|5.5 | -26.5 | 4.9 | -25.3|





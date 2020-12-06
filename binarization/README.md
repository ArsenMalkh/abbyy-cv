## Image Binarization

## Установка
```sh
python3 -m venv bin_env
source bin_env/bin/activate
pip3 install -r requirements.txt 
```


## Использование

```sh
$ python3 main.py -h
```

## Usage
For sauvola, 2018, wolf, skew
```
usage: main.py [-h] --imgs_path IMGS_PATH --save_dir SAVE_DIR
               [--block_size BLOCK_SIZE]
               [--binarization_type {sauvola,2018,wolf,skew}]

optional arguments:
  -h, --help            show this help message and exit
  --imgs_path IMGS_PATH
                        input images path
  --save_dir SAVE_DIR   save dir path
  --block_size BLOCK_SIZE
                        Block size for patterns.
  --binarization_type {sauvola,2018,wolf,skew}
                        type of binarization.

```
Были предприняты попытки улучшить результат с помощью денойзинга: лоупасс фильтра, повышения бинаризации.
Для улучшения был взят алгоритм пиромидальной бинаризации:
```
usage: pyramid_class.py [-h] --imgs_path IMGS_PATH --save_dir SAVE_DIR

optional arguments:
  -h, --help            show this help message and exit
  --imgs_path IMGS_PATH
                        input images path
  --save_dir SAVE_DIR   save dir path

```
Дополниттельно был взят константный сдвиг порога   на всех уровнях пирамиды. Метод в целом работает хорошо, но иногда шумит (в большей степени чем пресловутый Саувола, но зато хорошо выделяет светлые буквы на темном фоне)
### Результаты всех методов
Результаты лежат в папке data/results/


## Быстродействие:
Скорость выполнения программы для методов sauvola, 2018, wolf, skew с окном 50 на мегапиксель: 0.01s 

Для метода пирамидальной бинаризации 10s  на мегапиксель (python)



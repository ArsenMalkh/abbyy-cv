# Fractal compression task
## Creating environment
```sh
python3 -m venv debayering_env
source debayering_env/bin/activate
pip3 install -r requirements.txt 
```
## Usage
```
usage: main.py [-h] --img_path IMG_PATH --save_dir SAVE_DIR [--block_size {4,8}] [--n_iterations N_ITERATIONS] [--debug_scale DEBUG_SCALE]
optional arguments:
  -h, --help            show this help message and exit
  --img_path IMG_PATH   input image path
  --save_dir SAVE_DIR   save dir path
  --block_size {4,8}    Block size for patterns.
  --n_iterations N_ITERATIONS
                        Restoring iterations.
  --debug_scale DEBUG_SCALE
                        scale for debugging resize of original image
```

# Results
## Block size 4
### Lena
decoded image

![psnr](data/results/Lena_4/best.bmp)

Maximum psnr value: 32.08

![psnr](data/results/Lena_4/psnr.png)

### Boat
decoded image

![psnr](data/results/Boat_4/best.bmp)

Maximum psnr value:  29.72

![psnr](data/results/Boat_4/psnr.png)

### Goldhill
decoded image

![psnr](data/results/Goldhill_4/best.bmp)

Maximum psnr value: 30.91

![psnr](data/results/Goldhill_4/psnr.png)

## Block size 8
### Lena
decoded image

![psnr](data/results/Lena_8/best.bmp)

Maximum psnr value: 26.26

![psnr](data/results/Lena_8/psnr.png)

### Boat
decoded image

![psnr](data/results/Boat_8/best.bmp)

Maximum psnr value: 24.12

![psnr](data/results/Boat_8/psnr.png)

### Goldhill
decoded image

![psnr](data/results/Goldhill_8/best.bmp)

Maximum psnr value: 25.41

![psnr](data/results/Goldhill_8/psnr.png)




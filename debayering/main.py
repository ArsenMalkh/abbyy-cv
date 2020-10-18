import PIL.Image as Image
from src.ppg import PPG
from src.metrics import PSNR
from src.profile import profiler
import numpy as np
import time

source_image = Image.open("data/RGB_CFA.bmp")
source_np = np.array(source_image.convert('RGB'), dtype='int64')
original = Image.open("data/Original.bmp")
original_np =np.array(original.convert('RGB'), dtype='int64')
start_time = time.time()
output_np = PPG(source_np, original_np)
end_time = time.time()
whole_time, time_mp = profiler(start_time, end_time, source_np)


PSNR_value = PSNR(output_np, original_np)

output = Image.fromarray(output_np.astype('uint8'), mode='RGB')
output.save("data/answer.bmp")

print("PSNR {:.3f}, time for mp {:.3f} s, time for whole image {:.3f} s".format(PSNR_value, time_mp, whole_time))

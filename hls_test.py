import cv2
import numpy as np
import time


def main():
    original_image = cv2.imread('images/1.png',cv2.IMREAD_COLOR)
    low_image = cv2.resize(original_image,dsize=(original_image.shape[1] // 2, original_image.shape[0]//2))
    cv2.imwrite('low.jpg',low_image)
    lsr_image = cv2.resize(low_image,dsize=(original_image.shape[1], original_image.shape[0]))
    cv2.imwrite('lsr.jpg',low_image)
    lsr_hls = cv2.cvtColor(lsr_image,cv2.COLOR_BGR2HLS)
    original_hls = cv2.cvtColor(original_image,cv2.COLOR_BGR2HLS)
    
    lsr_hls[:,:,1] = original_hls[:,:,1]
    
    cv2.imwrite('lsr_ads.jpg',cv2.cvtColor(lsr_hls,cv2.COLOR_HLS2BGR))

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print("用时(s): ",end_time-start_time)
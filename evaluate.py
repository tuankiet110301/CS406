import streamlit as st
import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from colorattenuation import ColorAttenuationPrior  # Import Color Attenuation Prior Algorrithm
from darkchannel import DarkChannelPrior # Import Dark Channel Prior Algorrithm


def main():
    dehazer_color = ColorAttenuationPrior(0.85, 8)
    dehazer_dark = DarkChannelPrior()


    print("Color:\n")
    mse_color, ssim_color, avg_mse_color, avg_ssim_color, timerun = dehazer_color.evaluate_dehazing_algorithm()
    with open('color-indoor.txt', 'w') as file:
        file.write(f'MSE Color: {mse_color}\n')
        file.write(f'SSIM Color: {ssim_color}\n')
        file.write(f'Time: {timerun}\n')
        file.write(f'Average MSE Color: {avg_mse_color}\n')
        file.write(f'Average SSIM Color: {avg_ssim_color}\n')
    print("Dark:\n")
    mse_dark, ssim_dark, avg_mse_dark, avg_ssim_dark, timerun = dehazer_dark.evaluate_dehazing_algorithm()
    with open('dark-indoor.txt', 'w') as file:
        file.write(f'MSE Color: {mse_dark}\n')
        file.write(f'SSIM Color: {ssim_dark}\n')
        file.write(f'Time: {timerun}\n')
        file.write(f'Average MSE Color: {avg_mse_dark}\n')
        file.write(f'Average SSIM Color: {avg_ssim_dark}\n')


    #Save to evaluate.txt to display in main.py file

if __name__ == "__main__":
    main()







import cv2
import argparse
import numpy as np
import time
from skimage.metrics import structural_similarity as ssim
from scipy import signal as sig
from scipy.ndimage import minimum_filter, median_filter
import os

class ColorAttenuationPrior:
    def __init__(self, r_local_color=21, r_guided_color=7, gamma_value=1.0, beta=1.0):
        self.gamma_value = gamma_value
        self.r = r_local_color
        self.r_guided = r_guided_color
        self.beta = beta

    def guide(self, im, p, eps):
        mean_I = cv2.boxFilter(im, cv2.CV_64F, (self.r_guided, self.r_guided))
        mean_p = cv2.boxFilter(p, cv2.CV_64F, (self.r_guided, self.r_guided))
        mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (self.r_guided, self.r_guided))
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (self.r_guided, self.r_guided))
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = cv2.boxFilter(a, cv2.CV_64F, (self.r_guided, self.r_guided))
        mean_b = cv2.boxFilter(b, cv2.CV_64F, (self.r_guided, self.r_guided))

        q = mean_a * im + mean_b
        return q

    def localmin(self, D):
        R = self.r
        # Calculate the local minimum using scipy's minimum_filter
        LM = minimum_filter(D, size=(R, R))
        return LM


    # def hybrid_medfilt(self, image, window_size=5):
    #     # Square mask
    #     square = np.ones((window_size, window_size), dtype=bool)
    #
    #     # Cross mask
    #     cross = np.zeros((window_size, window_size), dtype=bool)
    #     cross[(window_size - 1) // 2, :] = True
    #     cross[:, (window_size - 1) // 2] = True
    #
    #     # Diag mask
    #     diag = np.zeros((window_size, window_size), dtype=bool)
    #     diag[np.arange(window_size), np.arange(window_size)] = True
    #     diag[np.arange(window_size), np.flip(np.arange(window_size))] = True
    #     if image.ndim == 2:
    #         h, w = image.shape
    #         hm = np.zeros((h, w), dtype=image.dtype)
    #         # print(f"H W C: {h}, {w}")
    #
    #         # Filtering
    #         hm1 = median_filter(image[:, :], size=(window_size, window_size), mode='reflect', footprint=square)
    #         hm2 = median_filter(image[:, :], size=(window_size, window_size), mode='reflect', footprint=cross)
    #         hm3 = median_filter(image[:, :], size=(window_size, window_size), mode='reflect', footprint=diag)
    #         temp = np.sort(np.dstack((hm1, hm2, hm3)), axis=-1)
    #         hm[:, :] = temp[:, :, 1]
    #
    #         return hm
    #     else:
    #         h, w, c = image.shape
    #         hm = np.zeros((h, w, c), dtype=image.dtype)
    #         # print(f"H W C: {h}, {w}")
    #
    #         # Filtering
    #         for index in range(c):
    #             hm1 = median_filter(image[:, :, index], size=(window_size, window_size), mode='reflect',
    #                                 footprint=square)
    #             hm2 = median_filter(image[:, :, index], size=(window_size, window_size), mode='reflect',
    #                                 footprint=cross)
    #             hm3 = median_filter(image[:, :, index], size=(window_size, window_size), mode='reflect', footprint=diag)
    #             temp = np.sort(np.dstack((hm1, hm2, hm3)), axis=-1)
    #             hm[:, :, index] = temp[:, :, 1]
    #
    #         return hm


    def postprocessing(self, GD, I, H, S, V):

        #Atmospheric Light Estimation
        flat_indices = np.argsort(GD, axis=None)
        top_indices_flat = flat_indices[int(np.round(0.999 * GD.size)):]
        top_indices = np.unravel_index(top_indices_flat, GD.shape)

        max_v_index = np.unravel_index(np.argmax(V[top_indices], axis=None), V.shape)
        I = I.astype(np.float32) / 255.0
        A = I[max_v_index[0], max_v_index[1], :]
        print(f"A: {A}\n")


        transmission = np.minimum(np.maximum(np.exp(-1 * self.beta * GD), 0.1), 0.9)

        # Convert the normalized image to grayscale
        grayscaleI = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)

        eps = 0.0001
        transmission2 = self.guide(grayscaleI, transmission, eps)
        print(f"Transmission {transmission}, grayscale I {grayscaleI}")
        transmission3 = np.full_like(I, transmission2[..., np.newaxis])
        # cv2.imshow('trans ', transmission3)
        # cv2.waitKey(0)
        # cv2.imshow('grayscale i ', grayscaleI)
        # cv2.waitKey(0)
        J = A + (I - A) / transmission3
        J = (J - np.min(J)) / (np.max(J) - np.min(J))
        J = (J * 255).astype(np.uint8)

        lut = (np.arange(256) / 255.0) ** (1.0 / self.gamma_value)
        J = cv2.LUT(J, lut)

        return J, transmission2

    def dehaze_image(self, input_path, output_path):
        start = time.time()
        I = cv2.imread(input_path)
        H, S, V = cv2.split(cv2.cvtColor(I, cv2.COLOR_RGB2HSV))
        V = V / 255.0
        S = S / 255.0

        theta_0, theta_1, theta_2 = 0.121779, 0.959710, -0.780245
        sigma = 0.041337
        epsilon = np.random.normal(0, sigma, H.shape)
        D = theta_0 + theta_1 * V + theta_2 * S + epsilon
        # cv2.imshow('Depth raw', D)
        # cv2.waitKey(0)
        output_image_D = output_path.split('.')[0] + '_depth.jpg'
        cv2.imwrite(output_image_D, D*255)


        LMD = self.localmin(D)
        # cv2.imshow('Depth after local min', LMD)
        # cv2.waitKey(0)
        output_image_D_local = output_path.split('.')[0] + '_depth_local.jpg'
        cv2.imwrite(output_image_D_local, LMD * 255)


        eps = 0.0001
        GD = self.guide(D, LMD, eps)
        # cv2.imshow('Depth after guided', GD)
        # cv2.waitKey(0)
        output_image_D_refine = output_path.split('.')[0] + '_depthrefine.jpg'
        cv2.imwrite(output_image_D_refine, (GD * 255))

        # cv2.imshow('Depth', GD)
        # cv2.waitKey(0)
        #GD = self.hybrid_medfilt(D, 5)

        # Remember to fix D -> GD below
        J, t = self.postprocessing(GD, I, H, S, V)

        #r, eps = 8, 0.2 * 0.2

        cv2.imwrite(output_path, J * 255)
        output_image_transmission = output_path.split('.')[0] + '_map.jpg'
        image_array = np.array(t*255, dtype=np.uint8)
        cv2.imwrite(output_image_transmission, image_array)
        end = time.time()
        execution_time = end - start
        return (J * 255).astype(np.uint8), (t * 255).astype(np.uint8), execution_time

    def dehaze_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(3))
        height = int(cap.get(4))
        fps = cap.get(5)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            i += 1
            print(f"Frame {i}\n")
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            H, S, V = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
            V = V / 255.0
            S = S / 255.0

            theta_0, theta_1, theta_2 = 0.121779, 0.959710, -0.780245
            sigma = 0.041337
            epsilon = np.random.normal(0, sigma, H.shape)
            D = theta_0 + theta_1 * V + theta_2 * S + epsilon

            LMD = self.localmin(D)
            eps = 0.0001
            GD = self.guide(D, LMD, eps)
            # Remember to fix D -> GD below
            J = self.postprocessing(GD, frame_rgb, H, S, V)

            J_bgr = cv2.cvtColor((J * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            out.write(J_bgr)

        cap.release()
        out.release()
        cv2.destroyAllWindows()


    def dehaze(self, input_path, output_path):
        # Check if input is an image or a video based on file extension
        _, extension = os.path.splitext(input_path.lower())

        if extension in ['.jpg', '.jpeg', '.png']:
            self.dehaze_image(input_path, output_path)
        elif extension == '.mp4':
            self.dehaze_video(input_path, output_path)
        else:
            print("Unsupported file format.")

    def evaluate_dehazing_algorithm(self, hazy_folder='datatest/hazy', gt_folder='datatest/GT'):
        hazy_images = sorted(os.listdir(os.path.join(hazy_folder)))
        gt_images = sorted(os.listdir(os.path.join(gt_folder)))

        if len(hazy_images) != len(gt_images):
            print("Error: Number of hazy images does not match the number of ground truth images.")
            return

        total_mse = []
        total_ssim = []
        timerun = []
        for hazy_image, gt_image in zip(hazy_images, gt_images):
            hazy_path = os.path.join(hazy_folder, hazy_image)
            gt_path = os.path.join(gt_folder, gt_image)
            print(gt_path)
            outputpath = 'result-indoor/color/' + os.path.basename(hazy_path)
            # Dehaze the hazy image
            dehazed_image, _, execution_time = self.dehaze_image(hazy_path, outputpath)

            # Load the ground truth image
            gt = cv2.imread(gt_path)

            # Evaluate metrics
            # Calculate squared differences for each channel
            squared_diff = (gt - dehazed_image) ** 2

            # Calculate the mean squared difference for each channel
            mean_squared_diff = np.mean(squared_diff, axis=(0, 1))

            # Calculate the overall MSE by taking the average across channels
            mse = np.sqrt(np.mean(mean_squared_diff))
            print(f"Time: {execution_time}")
            print(f"MSE: {mse}")
            print(gt.shape, dehazed_image.shape)
            ssim_value = ssim(gt, dehazed_image, win_size=7, channel_axis=-1)
            print(f"SSIM: {ssim_value}")

            timerun.append(execution_time)
            total_mse.append(mse)
            total_ssim.append(ssim_value)

        # Calculate average metrics
        avg_mse = sum(total_mse) / len(hazy_images)
        avg_ssim = sum(total_ssim) / len(hazy_images)
        return total_mse, total_ssim, avg_mse, avg_ssim, timerun

#if __name__ == "__main__":
#   parser = argparse.ArgumentParser(description="Dehaze an image or video.")
#   parser.add_argument("input_path", help="Path to the input image or video file")
#   parser.add_argument("output_path", help="Path to the output dehazed file")
#   parser.add_argument("--gamma_value", type=float, default=0.85, help="Gamma value for post-processing")
#   parser.add_argument("-r", type=int, default=8, help="Value of r for dehazing")
#   args = parser.parse_args()

dehazer = ColorAttenuationPrior()
dehazer.dehaze_image('trainin.jpg', 'trainininreal.jpg')
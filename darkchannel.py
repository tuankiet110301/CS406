import cv2
import numpy as np
import os
import time
from skimage.metrics import structural_similarity as ssim

class DarkChannelPrior:
    def __init__(self, r_dark=15, r_guided_dark=7):
        self.r_dark = r_dark
        self.r_guided = r_guided_dark

    def DarkChannel(self, im):
        b, g, r = cv2.split(im)
        dc = cv2.min(cv2.min(r, g), b)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.r_dark, self.r_dark))
        dark = cv2.erode(dc, kernel)
        # print(f"Dark Channel {dark}")
        # cv2.imshow('Dark Channel ', dark)
        # cv2.waitKey(0)
        return dark

    def AtmLight(self, im, dark):
        [h, w] = im.shape[:2]
        imsz = h * w
        numpx = int(max(np.floor(imsz / 100000), 1))
        darkvec = dark.reshape(imsz, 1)
        imvec = im.reshape(imsz, 3)
        print(f' Imvect: {imvec}')
        indices = darkvec.argsort()
        indices = indices[imsz - numpx ::]
        atmsum = np.zeros([1, 3])
        for ind in range(1, numpx):
            atmsum = atmsum + imvec[indices[ind]]
            print(f'Indices {imvec[indices[ind]]}')
        A = atmsum / (numpx-1)
        return A

    def TransmissionEstimate(self, im, A):
        omega = 0.95
        im3 = np.empty(im.shape, im.dtype)
        for ind in range(0, 3):
            im3[:, :, ind] = im[:, :, ind] / A[0, ind]
        dc = self.DarkChannel(im3)
        transmission = 1 - omega * dc
        return transmission

    def Guidedfilter(self, im, p, eps):
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


    def TransmissionRefine(self, im, et):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        gray = gray.astype("float64") / 255
        print(f"Gray {gray}")
        # cv2.imshow('Gray', gray)
        # cv2.waitKey(0)
        eps = 0.001
        t = self.Guidedfilter(gray, et, eps)
        return t

    def Recover(self, im, t, A, tx=0.1):
        res = np.empty(im.shape, im.dtype)
        t = np.maximum(t, tx)
        for ind in range(0, 3):
            res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]
        return res

    def dehaze_image(self, input_image_path, output_image_path):
        start = time.time()
        src = cv2.imread(input_image_path)
        I = src.astype("float64") / 255
        dark = self.DarkChannel(I)
        A = self.AtmLight(I, dark)
        print(f"A: {A}")
        te = self.TransmissionEstimate(I, A)

        # cv2.imshow('TE:', te)
        # cv2.waitKey(0)
        t = self.TransmissionRefine(src, te)

        # cv2.imshow('TE refine:', t)
        # cv2.waitKey(0)
        J = self.Recover(I, t, A, 0.1)

        cv2.imwrite(output_image_path, J*255)

        output_image_transmission = output_image_path.split('.')[0] + '_map.jpg'
        t_refined_save = np.array(t*255, dtype=np.uint8)
        cv2.imwrite(output_image_transmission, t_refined_save)



        end = time.time()
        execution_time = end - start
        return (J*255).astype(np.uint8), (t*255).astype(np.uint8), execution_time

    def dehaze_video(self, input_video_path, output_video_path):
        cap = cv2.VideoCapture(input_video_path)
        width = int(cap.get(3))
        height = int(cap.get(4))
        fps = cap.get(5)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        f = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            f += 1
            I = frame.astype("float64") / 255
            dark = self.DarkChannel(I)
            A = self.AtmLight(I, dark)
            te = self.TransmissionEstimate(I, A)
            t = self.TransmissionRefine(frame, te)
            J = self.Recover(I, t, A, 0.1)

            # Clip pixel values to ensure they are in the valid range [0, 255]
            J = np.clip(J * 255, 0, 255).astype(np.uint8)

            # Convert the dehazed frame back to BGR
            #J_bgr = cv2.cvtColor(J, cv2.COLOR_RGB2BGR)

            out.write(J)

        cap.release()
        out.release()
        cv2.destroyAllWindows()

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
            outputpath = 'result-indoor/dark/' + os.path.basename(hazy_path)
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
            #print(f"Image: {hazy_image}, MSE: {mse}, SSIM: {ssim_value}")

            total_mse.append(mse)
            total_ssim.append(ssim_value)
            timerun.append(execution_time)
        # Calculate average metrics
        avg_mse = sum(total_mse) / len(hazy_images)
        avg_ssim = sum(total_ssim) / len(hazy_images)
        return total_mse, total_ssim, avg_mse, avg_ssim, timerun






if __name__ == "__main__":
   dehazer = DarkChannelPrior()
   input_filename = "cit.png"
   output_filename = "cit_out.jpg"
   dehazer.dehaze_image('1859.jpg', '1859-out.jpg')

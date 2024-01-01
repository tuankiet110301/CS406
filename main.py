import streamlit as st
import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from colorattenuation import ColorAttenuationPrior  # Import Color Attenuation Prior Algorrithm
from darkchannel import DarkChannelPrior # Import Dark Channel Prior Algorrithm

class DehazerApp:
    def __init__(self, r_dark, r_guided_dark, r_local_color, r_guided_color, gamma_value, beta):
        self.dehazer_color_attenuation = ColorAttenuationPrior(r_local_color, r_guided_color, gamma_value, beta)
        self.dehazer_dark_channel = DarkChannelPrior(r_dark, r_guided_dark)

    def dehaze_image(self, input_file, color_out, dark_out):
        #im = cv2.imread(input_file)

        # Dehaze using color attenuation
        cap, trans_cap, time_cap = self.dehazer_color_attenuation.dehaze_image(input_file, color_out)

        # Dehaze using dark channel
        dcp, trans_dcp, time_dcp = self.dehazer_dark_channel.dehaze_image(input_file, dark_out)

        return cap, trans_cap, time_cap, dcp, trans_dcp, time_dcp

    # def get_metrics_color(self):
    #     return self.mse_color, self.ssim_color, self.avg_mse_color, self.avg_ssim_color
    #
    # def get_metris_dark(self):
    #     return self.mse_dark, self.ssim_dark, self.avg_mse_dark, self.avg_ssim_dark


def display_results(title, origin, dehazed, transmission_map, execution_time):
    st.write(f"### {title}")
    st.image([origin, dehazed, transmission_map], caption=['Origin', 'Dehazed', 'Transmission Map'], width=350)
    st.write(f"Execution Time: {execution_time:.2f} seconds")

def main():
    st.title("Image Dehazing App")

    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file)

        #DCP
        r_dark = st.slider("Dark patch", min_value=1, max_value=200, value=15, step=1)
        r_guided_dark = st.slider("Guided filter patch (DCP)", min_value=1, max_value=200, value=7, step=1)

        #CAP
        r_local_color = st.slider("Local minimum patch", min_value=1, max_value=200, value=15, step=1)
        r_guided_color = st.slider("Guided filter patch (CAP)", min_value=1, max_value=200, value=7, step=1)
        gamma_value = st.slider("Gamma Value", min_value=0.1, max_value=2.5, value=1.0, step=0.05)
        beta = st.slider("Beta Value", min_value=0.1, max_value=2.5, value=1.0, step=0.05)

        # Save input to local path
        file_path = os.path.join('input2.jpg')

        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())


        dehazer_app = DehazerApp(r_dark, r_guided_dark, r_local_color, r_guided_color, gamma_value, beta)
        color_out = 'images/color/test.jpg'
        dark_out = 'images/dark/test.jpg'


        if st.button("Dehaze"):
            #if uploaded_file.type in ["jpeg", "png", "jpg"]:
            print("Success")
            cap, trans_cap, time_cap, dcp, trans_dcp, time_dcp = dehazer_app.dehaze_image(file_path, color_out, dark_out)
            bgr_cap = cv2.cvtColor(cap, cv2.COLOR_RGB2BGR)
            #bgr_dcp = cv2.cvtColor(dcp, cv2.COLOR_RGB2BGR)

            dcp = cv2.imread('images/dark/test.jpg')
            bgr_dcp = cv2.cvtColor(dcp, cv2.COLOR_RGB2BGR)

            display_results("CAP", uploaded_file, bgr_cap, trans_cap, time_cap)
            display_results("DCP", uploaded_file, bgr_dcp, trans_dcp, time_dcp)


if __name__ == "__main__":
    main()
# Class containing utility functions for the project

#import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
import cv2


base_path = "../data/"
save_path = "/Users/conorosullivan/Google Drive/My Drive/UCD/research/Journal Paper 2 - superpixels/figures/"

# dictionary of band names for each satellite
band_dic = {"sentinel":{"coastal":0,"blue":1,"green":2,"red":3,"rededge1":4,"rededge2":5,"rededge3":6,"nir":7,"narrownir":8,"watervapour":9,"swir1":10,"swir2":11},
         "landsat":{"blue":0,"green":1,"red":2,"nir":3,"swir1":4,"swir2":5,"thermal":6}}


def save_fig(fig, name):
    """Save figure to figures folder"""
    fig.savefig(save_path + f"/{name}.png", dpi=300, bbox_inches="tight")


def display_bands(img, satellite="landsat"):
    """Visualize all SR bands of a satellite image."""

    if satellite == "landsat":
        n = img.shape[2]
        band_names = ["Blue","Green","Red","NIR","SWIR1","SWIR2","Thermal"]
        if n == 8:
            band_names.append("Mask")
        else:
            # Bands for LICS test set 
            band_names.extend(["QA","Train","Mask","Edge"])

    elif satellite == "sentinel":
        n = 13
        band_names = [ "Coastal","Blue","Green","Red","Red Edge 1","Red Edge 2","Red Edge 3","NIR","Red Edge 4","Water Vapour","SWIR1","SWIR2","Mask"]

    fig, axs = plt.subplots(1, n, figsize=(20, 5))

    for i in range(n):
        if np.unique(img[:, :, i]).size == 1:
            axs[i].imshow(img[:, :, i], cmap="gray", vmin=0, vmax=1)
        else:
            axs[i].imshow(img[:, :, i], cmap="gray")
        axs[i].set_title(band_names[i])
        axs[i].axis("off")


def scale_bands(img,satellite="landsat"):
    """Scale bands to 0-1"""
    img = img.astype("float32")
    if satellite == "landsat":
        img = np.clip(img * 0.0000275 - 0.2, 0, 1)
    elif satellite == "sentinel":
        img = np.clip(img/10000, 0, 1)
    return img

def edge_from_mask(mask):
    """Get edge map from mask"""

    dy, dx = np.gradient(mask)
    grad = np.abs(dx) + np.abs(dy)
    edge = np.array([grad > 0])[0]
    edge = edge.astype(np.uint8)

    return edge


def get_rgb(img, bands=['red','green',"blue"],satellite ='landsat', contrast=1):
    """Convert a stacked array of bands to RGB"""

    r = band_dic[satellite][bands[0]]
    g = band_dic[satellite][bands[1]]
    b = band_dic[satellite][bands[2]]

    rgb = img[:,:, [r,g,b]]
    rgb = rgb.astype(np.float32)
    rgb = scale_bands(rgb, satellite)
    rgb = np.clip(rgb, 0, contrast) / contrast

    return rgb

def histogram_equalization(img):
    """Apply histogram equalization to an image"""

    # Rescale the image to 0-255 and convert to uint8
    if img.max() <= 1:
        img_rescaled = (img * 255).astype(np.uint8)

    # Split the rescaled image into its respective channels (R, G, B)
    R, G, B = cv2.split(img_rescaled)

    # Apply histogram equalization to each channel
    R = cv2.equalizeHist(R)
    G = cv2.equalizeHist(G)
    B = cv2.equalizeHist(B)

    # Merge the equalized channels back together
    img_equalised = cv2.merge((R,G,B))
    
    return img_equalised

def histogram_equalization_luminance(img):
    """Apply histogram equalization to the luminance of an image"""
    
    # Rescale the image to 0-255 if necessary and convert to uint8
    if img.max() <= 1:
        img_rescaled = (img * 255).astype(np.uint8)
    else:
        img_rescaled = img.astype(np.uint8)
    
    # Convert the image from RGB to YCrCb color space
    img_YCrCb = cv2.cvtColor(img_rescaled, cv2.COLOR_RGB2YCrCb)
    
    # Split the image into its respective channels (Y, Cr, Cb)
    Y, Cr, Cb = cv2.split(img_YCrCb)
    
    # Apply histogram equalization to the luminance channel (Y)
    Y_equalized = cv2.equalizeHist(Y)
    
    # Merge the channels back together
    img_YCrCb_equalized = cv2.merge((Y_equalized, Cr, Cb))
    
    # Convert the image back to RGB color space
    img_equalized = cv2.cvtColor(img_YCrCb_equalized, cv2.COLOR_YCrCb2RGB)
    
    return img_equalized


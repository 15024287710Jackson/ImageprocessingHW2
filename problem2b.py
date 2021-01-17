import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import iradon,radon

if __name__ == '__main__':
    image = np.zeros((100,100))
    image[10:50,20:25]=255
    plt.figure(1)
    plt.title('Original image')
    plt.imshow(image, cmap='gray');
    theta_45 = np.linspace(0., 180., 4, endpoint=False)
    fourtyfive_image=radon(image, theta=theta_45, circle=True)
    plt.figure(2)
    plt.title('fourtyfive_sinogram')
    plt.imshow(fourtyfive_image, cmap='gray');
    theta_18 = np.linspace(0., 180., 18, endpoint=False)
    ten_image = radon(image, theta=theta_18, circle=True)
    plt.figure(3)
    plt.title('ten_sinogram')
    plt.imshow(ten_image, cmap='gray');
    theta_5 = np.linspace(0., 180., 36, endpoint=False)
    five_image = radon(image, theta=theta_5, circle=True)
    plt.figure(4)
    plt.title('five_sinogram')
    plt.imshow(five_image, cmap='gray');
    theta_2 = np.linspace(0., 180., 90, endpoint=False)
    two_image = radon(image, theta=theta_2, circle=True)
    plt.figure(5)
    plt.title('two_sinogram')
    plt.imshow(two_image, cmap='gray');
    theta_1 = np.linspace(0., 180., 180, endpoint=False)
    one_image = radon(image, theta=theta_1, circle=True)
    plt.figure(6)
    plt.title('one_sinogram')
    plt.imshow(one_image, cmap='gray');
    reconfbp_hamming_45 = iradon(fourtyfive_image, theta=theta_45, circle=True, filter_name='hamming')
    reconfbp_hamming_18 = iradon(ten_image, theta=theta_18, circle=True, filter_name='hamming')
    reconfbp_hamming_5 = iradon(five_image, theta=theta_5, circle=True, filter_name='hamming')
    reconfbp_hamming_2 = iradon(two_image, theta=theta_2, circle=True, filter_name='hamming')
    reconfbp_hamming_1 = iradon(one_image, theta=theta_1, circle=True, filter_name='hamming')
    plt.figure(7)
    plt.title('reconstruction_fbp_hamming_fourtyfive_image')
    plt.imshow(reconfbp_hamming_45, cmap='gray');
    plt.figure(8)
    plt.title('reconstruction_fbp_hamming_ten_image')
    plt.imshow(reconfbp_hamming_18, cmap='gray');
    plt.figure(9)
    plt.title('reconstruction_fbp_hamming_five_image')
    plt.imshow(reconfbp_hamming_5, cmap='gray');
    plt.figure(10)
    plt.title('reconstruction_fbp_hamming_two_image')
    plt.imshow(reconfbp_hamming_2, cmap='gray');
    plt.figure(11)
    plt.title('reconstruction_fbp_hamming_one_image')
    plt.imshow(reconfbp_hamming_1, cmap='gray');
    reconbp_45 = iradon(fourtyfive_image, theta=theta_45, circle=True, filter_name=None)
    reconbp_18 = iradon(ten_image, theta=theta_18, circle=True, filter_name=None)
    reconbp_5 = iradon(five_image, theta=theta_5, circle=True, filter_name=None)
    reconbp_2 = iradon(two_image, theta=theta_2, circle=True, filter_name=None)
    reconbp_1 = iradon(one_image, theta=theta_1, circle=True, filter_name=None)
    plt.figure(12)
    plt.title('reconstruction_simple(nonfiltered)back-projection_fourtyfive_image')
    plt.imshow(reconbp_45, cmap='gray')
    plt.figure(13)
    plt.title('reconstruction_simple(nonfiltered)back-projection_ten_image')
    plt.imshow(reconbp_18, cmap='gray')
    plt.figure(14)
    plt.title('reconstruction_simple(nonfiltered)back-projection_five_image')
    plt.imshow(reconbp_5, cmap='gray')
    plt.figure(15)
    plt.title('reconstruction_simple(nonfiltered)back-projection_two_image')
    plt.imshow(reconbp_2, cmap='gray')
    plt.figure(16)
    plt.title('reconstruction_simple(nonfiltered)back-projection_one_image')
    plt.imshow(reconbp_1, cmap='gray')
    plt.show()
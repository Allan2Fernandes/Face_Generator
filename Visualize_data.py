import matplotlib.pyplot as plt

def display_single_image(image_array):
    plt.imshow(image_array*0.5 + 0.5)
    plt.show()

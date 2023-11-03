from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
from PIL import Image
from pathlib import Path
import glob

def animate(frame): 
    im.set_data(img_array[frame])
    return im,

def extract_number_from_filename(filename: str) -> int:
    """Extract the number from the filename."""
    return int(filename.split("plot_mppi_")[1].split(".png")[0])




if __name__ == '__main__': 

    path = str(Path(__file__).parents[0]) + '/plots/'

    files = sorted(glob.glob(path + '*.png'), key=extract_number_from_filename)

    img_array = [Image.open(file) for file in files]

    fig, ax = plt.subplots()

    im = ax.imshow(img_array[0], animated=True)

    animation_fig = FuncAnimation(fig, animate, frames=len(img_array), interval=100, blit=True)

    animation_fig.save(path + 'animation_new.gif')




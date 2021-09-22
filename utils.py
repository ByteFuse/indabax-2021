from matplotlib import pyplot as plt
from IPython.display import clear_output

def render(env):
    img = env.render(mode='rgb')
    plt.imshow(img, interpolation='nearest')
    clear_output(wait = True)
    plt.pause(0.00001)
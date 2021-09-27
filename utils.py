from matplotlib import pyplot as plt
from IPython.display import clear_output

def render(env, mode='rgb'):
    img = env.render(mode=mode)
    plt.imshow(img, interpolation='nearest')
    clear_output(wait = True)
    plt.pause(0.00001)
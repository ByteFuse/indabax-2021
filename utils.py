from matplotlib import pyplot as plt
from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


def render(env, mode='rgb'):
    img = env.render(mode=mode, highlight=False)
    plt.imshow(img, interpolation='nearest')
    clear_output(wait=True)
    plt.pause(0.00001)


def draw_triangle(p1=[1, 1], p2=[2, 2.5], p3=[3, 1], c='blue', alpha=0.3):
    X = np.array([p1, p2, p3])
    Y = [c, c, c]
    plt.scatter(X[:, 0], X[:, 1], s=0.001, color=Y[:])
    t1 = plt.Polygon(X[:3, :], color=Y[0], alpha=alpha)
    plt.gca().add_patch(t1)


def draw_square(lb_x,
                lb_y,
                size,
                alpha=1,
                l_color='green',  # left
                r_color='yellow',  # right
                b_color='red',  # down
                t_color='blue'):  # up
    tr_x = lb_x+size  # top right x
    tr_y = lb_y+size  # top right y

    md_x = lb_x + size/2  # middle x
    md_y = lb_y + size/2  # middle x

    width = 10

    md_x_plus = md_x + width/2
    md_x_minus = md_x - width/2

    md_y_plus = md_y + width/2
    md_y_minus = md_y - width/2

    colormap = cm.get_cmap('RdYlGn')

    color_right = colormap(r_color)
    color_down = colormap(b_color)
    color_left = colormap(l_color)
    color_up = colormap(t_color)

    draw_triangle([lb_x+2, md_y], [md_x_minus, md_y_minus+2],
                  [md_x_minus, md_y_plus-2], c=color_left, alpha=alpha)  # left
    draw_triangle([tr_x-2, md_y], [md_x_plus, md_y_minus+2],
                  [md_x_plus, md_y_plus-2], c=color_right, alpha=alpha)  # right
    draw_triangle([md_x, tr_y-2], [md_x_minus+2, md_y_plus],
                  [md_x_plus-2, md_y_plus], c=color_down, alpha=alpha)  # down
    draw_triangle([md_x, lb_y+2], [md_x_minus+2, md_y_minus],
                  [md_x_plus-2, md_y_minus], c=color_up, alpha=alpha)  # up

from matplotlib import pyplot as plt
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np

def render(env, mode='rgb'):
    img = env.render(mode=mode)
    plt.imshow(img, interpolation='nearest')
    clear_output(wait = True)
    plt.pause(0.00001)
    
def draw_triangle(p1=[1,1], p2=[2,2.5], p3= [3, 1], c='blue', alpha=0.3):
    X = np.array([p1, p2, p3])
    Y = [c,c,c]
    plt.scatter(X[:, 0], X[:, 1], s= 0.001, color = Y[:])
    t1 = plt.Polygon(X[:3,:], color=Y[0], alpha=alpha)
    plt.gca().add_patch(t1)
    
def draw_square(lb_x, lb_y, size, alpha=0.1, l_color='green', r_color='yellow', b_color='red', t_color='blue'):
    tr_x = lb_x+size  # top right x
    tr_y = lb_y+size  # top right y

    md_x = lb_x + size/2  # middle x
    md_y = lb_y + size/2  # middle x

    draw_triangle([lb_x, lb_y], [tr_x, lb_y], [md_x, md_y], c=t_color, alpha=alpha)
    draw_triangle([lb_x, tr_y], [tr_x, tr_y], [md_x, md_y], c=b_color, alpha=alpha)
    draw_triangle([lb_x, tr_y], [lb_x, lb_y], [md_x, md_y], c=l_color, alpha=alpha)
    draw_triangle([tr_x, tr_y], [tr_x, lb_y], [md_x, md_y], c=r_color, alpha=alpha)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import mitsuba as mi

def plot_texture_samples(x, coords):
    img = mi.util.convert_to_bitmap(x.mean(axis=0))
    img2 = mi.util.convert_to_bitmap(x.var(axis=0).sum(axis=-1))
    # img2 = x.var(axis=0).sum(axis=-1).sqrt().cpu().detach()
    n = len(coords)
    
    fig = plt.figure(figsize=(10.5, 4.5))
    gs = gridspec.GridSpec(n, 3, width_ratios=[3, 1, 3], wspace=0.03, hspace=0.1)
    cmap = plt.get_cmap('tab10')

    # Left subplot
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.imshow(img)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("Estimated texture")

    # Right subplot
    ax2 = fig.add_subplot(gs[:, 2])
    ax2.imshow(img2)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title("uncertainty map (std)")

    for row in range(n):
        a,b = coords[row]
        rectsize = 8
        rx = b - rectsize/2
        ry = a - rectsize/2
        rect = patches.Rectangle((rx, ry), rectsize, rectsize, linewidth=1.1, edgecolor=cmap(row), facecolor='none')
        ax1.add_patch(rect)
        
        ax = fig.add_subplot(gs[row, 1])
        pts = x[:,a,b,1::2].cpu().detach()
        ax.scatter(pts[:,0], pts[:,1], s=5, c='black')
        for spine in ['left', 'right', 'top', 'bottom']:
            ax.spines[spine].set_color(cmap(row))
            ax.spines[spine].set_linewidth(1.1) 
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('square')
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        if row == 0:
            ax.set_title("texel distribution")
    plt.tight_layout()
    plt.show()



def animate_particles(particles, title="particles"):
    # particles: (frame, particles, dims)
    assert particles.dim() == 3 and particles.shape[-1] == 2
    fig, ax = plt.subplots(figsize=(4, 4))
    scatter = ax.scatter(particles[0, :, 0], particles[0, :, 1], s=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    # frame callback
    def update(frame):
        scatter.set_offsets(particles[frame])
        return scatter,
    # Create the animation. interval = ms per frame
    ani = FuncAnimation(fig, update, frames=particles.shape[0], blit=True, interval=30)
    # Convert the animation to HTML5 video format for displaying in the notebook
    plt.close()
    return HTML(ani.to_jshtml())
    
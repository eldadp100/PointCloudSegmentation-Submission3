from matplotlib import pyplot as plt

from try_with_pointnet.load_data import get_data


def plot_3d_on_ax(ax, point_cloud_pos, y):
    xs, ys, zs = point_cloud_pos[:, 0], point_cloud_pos[:, 1], point_cloud_pos[:, 2]
    ax.scatter(xs, ys, zs, c=y)


def plot_3d(point_cloud_pos, y, title="", show=True, path_to_save=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_3d_on_ax(ax, point_cloud_pos, y)
    plt.show()


def plot_3d_batch_at_index(batch, index, title="", show=True, path_to_save=None):
    y = batch.y[batch.batch == index]
    pos = batch.pos[batch.batch == index]
    plot_3d(pos, y, title, show=True, path_to_save=None)


def plot_3d_grid(rows, cols, data):
    fig, ax = plt.subplot(nrows=rows, ncols=cols)
    for i in range(rows):
        for j in range(cols):
            plot_3d_on_ax(ax[i][j], data[i][j]["pos"], data[i][j]["y"])
    plt.show()


# visualize some samples
if __name__ == '__main__':
    dl_train, dl_test = get_data(32)
    plot_3d_batch_at_index(next(dl_train.__iter__()), 25)

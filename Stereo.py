import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D              
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_x (fx, ul, ox, z):
    X = (ul-ox)*z/fx
    return X

def get_y (fy, vl, oy, z):
    Y = (vl-oy)*z/fy
    return Y

def get_z (b, ul, ur, fx):
    d = ul-ur
    if abs(d) <  1:
        return # here mean the value of disparity is too small, and we did valeur abs because the disparity must be positive not negayive  
    Z = b*fx/abs(d)
    if Z <= 0 or Z > 5000:
        return None
    return Z 

def get_positions (path_folder):
    x1 = np.loadtxt(f"{path_folder}/x1_position.txt")
    y1 = np.loadtxt(f"{path_folder}/y1_position.txt")
    x2 = np.loadtxt(f"{path_folder}/x2_position.txt")
    y2 = np.loadtxt(f"{path_folder}/y2_position.txt")
    return x1, y1, x2, y2

# to get la matrice k that has la focal and so on 
def get_k(path_folder):
    return np.load(f"camera_parameters/mtx.npy")


def reconstruct_3d (path_folder, b):
    k = get_k(path_folder)
    fx = k[0,0]
    fy = k[1,1]
    ox = k[0,2]
    oy = k[1,2]

    x1_list, y1_list, x2_list, y2_list = get_positions(path_folder)

    points_3d = []
    rejected = 0
    for ul, vl, ur, vr in zip(x1_list, y1_list, x2_list, y2_list):
        Z = get_z(b,ul,ur,fx)

        if Z is None: 
            rejected += 1
        else:
          X = get_x(fx,ul,ox,Z)
          Y = get_y(fy,vl,oy,Z)
          points_3d.append([X, Y, Z])

    pts = np.array(points_3d)
    print(f"Reconstructed : {len(pts)} points  |  rejected : {rejected}")

    # if len(pts) > 10:
    #     q1, q3 = np.percentile(pts[:, 2], [10, 90])
    #     iqr     = q3 - q1
    #     mask    = (pts[:, 2] >= q1 - 1.5 * iqr) & (pts[:, 2] <= q3 + 1.5 * iqr)
    #     removed = np.sum(~mask)
    #     pts     = pts[mask]
    #     print(f"After outlier removal : {len(pts)} points  (removed {removed})")

    return pts

# visualization 

def visualize_3d (points_3d):
    if len(points_3d) == 0:
        print("no 3d points to display")
        return
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # z_value = points_3d[:,2]
    scatter = ax.scatter(
        points_3d[:,0],
        points_3d[:,1],
        points_3d[:,2],
        c = points_3d[:, 2], 
        cmap = "plasma", 
        s = 12,
        depthshade=True
    )
    plt.colorbar(scatter, ax=ax, label='Profondeur Z (mm)', shrink=0.6)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z profondeur (mm)')
    plt.title(f'Nuage de points 3D — {len(points_3d)} points')
    plt.tight_layout()
    plt.show()


def visualize_mesh(points_3d: np.ndarray, n_clusters: int = 3):
    if len(points_3d) < n_clusters * 4:
        print("Not enough points for clustered mesh.")
        visualize_3d(points_3d)
        return

    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = kmeans.fit_predict(points_3d[:, 2].reshape(-1, 1))

    cluster_means = [(labels == i, points_3d[labels == i, 2].mean()) for i in range(n_clusters)]
    cluster_means.sort(key=lambda x: x[1])

    colors    = ['steelblue', 'tomato', 'gold']
    edge_cols = ['navy',      'darkred', 'goldenrod']

    fig = plt.figure(figsize=(14, 9))
    ax  = fig.add_subplot(111, projection='3d')

    for idx, (mask, _) in enumerate(cluster_means):
        cluster_pts = points_3d[mask]
        print(f"Cluster {idx+1}: {len(cluster_pts)} points, "
              f"Z mean={cluster_pts[:,2].mean():.0f} mm")

        if len(cluster_pts) < 4:
            ax.scatter(*cluster_pts.T, color=colors[idx], s=18)
            continue

        try:
            hull = ConvexHull(cluster_pts)
        except Exception as e:
            print(f"  ConvexHull failed for cluster {idx+1}: {e}")
            ax.scatter(*cluster_pts.T, color=colors[idx], s=18)
            continue

        triangles = [cluster_pts[simplex] for simplex in hull.simplices]
        poly = Poly3DCollection(
            triangles,
            alpha=0.30,
            facecolor=colors[idx],
            edgecolor=edge_cols[idx],
            linewidth=0.4,
        )
        ax.add_collection3d(poly)
        ax.scatter(*cluster_pts.T, color=colors[idx], s=10, zorder=5)

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    plt.title(f'Mesh 3D — {n_clusters} objects (clustered by depth)')
    plt.tight_layout()
    plt.show()


def visualize_boxes_from_known_dimensions(points_3d: np.ndarray):

    # (width_x, height_y, depth_z, name, color)
    # height_y is the TALL dimension — the box stands up in Y
    known_dims = {
        0: (40,  72, 138,  'Kayali Perfume',  'steelblue'),
        1: (36,  42, 107,  'Derma-T',  'tomato'),
        2: (31,  31, 112,  'Touché Serum',  'gold'),
    }

    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
    labels = kmeans.fit_predict(points_3d[:, 2].reshape(-1, 1))

    cluster_info = []
    for i in range(3):
        mask = labels == i
        pts  = points_3d[mask]
        cluster_info.append((pts[:, 2].mean(), i, pts))
    cluster_info.sort()  # sort by Z mean: nearest box first

    fig = plt.figure(figsize=(14, 9))
    ax  = fig.add_subplot(111, projection='3d')

    def draw_box(ax, cx, cy, cz, dx, dy, dz, color, label):
        """Draw an upright box. cy = bottom of box (max Y). Box grows upward (Y decreases)."""
        x0, x1 = cx - dx/2, cx + dx/2
        y0, y1 = cy, cy - dy        # y0=bottom, y1=top (Y axis points down in camera coords)
        z0, z1 = cz - dz/2, cz + dz/2

        corners = [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],  # front face
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],  # back face
        ]
        faces = [
            [corners[0], corners[1], corners[2], corners[3]],  # front
            [corners[4], corners[5], corners[6], corners[7]],  # back
            [corners[0], corners[1], corners[5], corners[4]],  # bottom
            [corners[2], corners[3], corners[7], corners[6]],  # top
            [corners[0], corners[3], corners[7], corners[4]],  # left
            [corners[1], corners[2], corners[6], corners[5]],  # right
        ]
        poly = Poly3DCollection(faces, alpha=0.25,
                                facecolor=color, edgecolor='gray', linewidth=0.6)
        ax.add_collection3d(poly)
        ax.text(cx, cy - dy - 5, cz, label, ha='center', fontsize=8)

    for rank, (zmean, cluster_idx, pts) in enumerate(cluster_info):
        cx = pts[:, 0].mean()
        cy = pts[:, 1].max()   # bottom of box = lowest point in cluster (max Y)
        cz = zmean
        dx, dy, dz, name, color = known_dims[rank]

        print(f"{name}: center=({cx:.0f}, {cy:.0f}, {zmean:.0f}) mm")
        ax.scatter(*pts.T, color=color, s=12, zorder=5, label=name)   # ← inside loop
        draw_box(ax, cx, cy, cz, dx, dy, dz, color, name)             # ← inside loop

    # These stay outside the loop
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.legend(loc='upper left', fontsize=9)
    plt.title('3D reconstruction — boxes fitted from known dimensions')
    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
import os


os.makedirs("figures", exist_ok=True)


phi = (1 + np.sqrt(5)) / 2


vertices = np.array([
    [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
    [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
    [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
])


vertices = vertices / np.linalg.norm(vertices[0])


faces = np.array([
    [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
    [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
    [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
])

def subdivide_face(v0, v1, v2, N):
    new_vertices = []
    new_faces = []
    vertex_map = {}  # карта: (i, j, k) -> index

    for i in range(N + 1):
        for j in range(N + 1 - i):
            k = N - i - j
            point = (i * v0 + j * v1 + k * v2) / N
            point = point / np.linalg.norm(point)
            new_vertices.append(point)
            vertex_map[(i, j, k)] = len(new_vertices) - 1

    for i in range(N):
        for j in range(N - i):
            k = N - i - j
            a = vertex_map[(i,     j,     k)]
            b = vertex_map[(i+1,   j,     k-1)]
            c = vertex_map[(i,     j+1,   k-1)]
            new_faces.append([a, b, c])
            if i + 1 <= N - 1 and j + 1 <= N - (i + 1):
                a = vertex_map[(i+1,   j,     k-1)]
                b = vertex_map[(i+1,   j+1,   k-2)]
                c = vertex_map[(i,     j+1,   k-1)]
                new_faces.append([a, b, c])

    return np.array(new_vertices), new_faces

def create_geodesic_dome(N):
    all_vertices = []
    all_faces = []
    vertex_dict = {}

    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        sub_verts, sub_faces = subdivide_face(v0, v1, v2, N)

        local_to_global = []
        for v in sub_verts:
            key = tuple(np.round(v, decimals=10))
            if key not in vertex_dict:
                vertex_dict[key] = len(all_vertices)
                all_vertices.append(v)
            local_to_global.append(vertex_dict[key])

        for tri in sub_faces:
            global_tri = [local_to_global[i] for i in tri]
            all_faces.append(global_tri)

    return np.array(all_vertices), np.array(all_faces)

def plot_sphere(ax, R=1.0, alpha=0.08):
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 15)
    x = R * np.outer(np.cos(u), np.sin(v))
    y = R * np.outer(np.sin(u), np.sin(v))
    z = R * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='gray', alpha=alpha, linewidth=0)

def plot_mesh(ax, vertices, faces, color='lightblue', alpha=0.7, edgecolor='k', linewidth=0.4):
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    tri_verts = [[vertices[face[0]], vertices[face[1]], vertices[face[2]]] for face in faces]
    collection = Poly3DCollection(tri_verts, facecolors=color, alpha=alpha, edgecolors=edgecolor, linewidths=linewidth)
    ax.add_collection3d(collection)
    max_range = 1.0
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])


models = {}
for N in [1, 2, 3]:
    verts, faces_out = create_geodesic_dome(N)
    models[N] = (verts, faces_out)


fig, axs = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={'projection': '3d'})
colors = {1: 'lightcoral', 2: 'lightgreen', 3: 'lightblue'}

for idx, N in enumerate([1, 2, 3]):
    ax = axs[idx]
    plot_sphere(ax)
    verts, faces_out = models[N]
    plot_mesh(ax, verts, faces_out, color=colors[N])
    ax.set_box_aspect([1,1,1])
    ax.set_axis_off()
    ax.set_title(rf'$N = {N}$ ({len(faces_out)} граней)')

plt.tight_layout()
plt.savefig("figures/comparison_N1_N2_N3.png", dpi=300, bbox_inches='tight')
plt.close()


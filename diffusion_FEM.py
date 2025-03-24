import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, lil_matrix
from scipy.sparse.linalg import spsolve


class Diffusion():
    def __init__(self, node=None, elem=None, nodes_Dirichlet=None, values_Dirichlet=None,
                 edges_Neumann=None, values_Neumann=None, material=None, source=None):
        self.node = node
        self.elem = elem
        self.nodes_Dirichlet = nodes_Dirichlet
        self.values_Dirichlet = values_Dirichlet

        if edges_Neumann is None:
            self.edges_Neumann = np.array([]).reshape(0, 2)  # empty Neumann boundary
            self.values_Neumann = np.array([])  # no Neumann conditions
        else:
            self.edges_Neumann = edges_Neumann
            self.values_Neumann = values_Neumann

        self.coords_x = self.node[:, 0]
        self.coords_y = self.node[:, 1]
        self.n_node = self.node.shape[0]
        self.n_elem = self.elem.shape[0]
        self.centers = np.column_stack((np.mean(self.coords_x[self.elem], axis=1), np.mean(self.coords_y[self.elem], axis=1)))

        self.material = material

        # if source is a scalar, set to constant source value
        if isinstance(source, int):
            self.source = np.full(self.n_elem, source)
        # if source is a list, set to corresponding source values
        elif isinstance(source, list):
            self.source = np.array(source)
        # if source is a numpy array, set to corresponding source values
        elif isinstance(source, np.ndarray):
            self.source = source
        # if source is a function, compute source values for each element center
        elif callable(source):
            self.source = source(self.centers)
        # if source is None, set to constant 0 (default)
        else:
            self.source = np.zeros(self.n_elem)

        self._prepare()

    def _prepare(self):
        # extract vertex coordinates for each triangle (triangles: (ntri, 3))
        self.p1 = self.node[self.elem[:, 0]]
        self.p2 = self.node[self.elem[:, 1]]
        self.p3 = self.node[self.elem[:, 2]]

        # compute the area of triangles given vertex coordinates
        self.areas = 0.5 * np.abs((self.p2[:, 0] - self.p1[:, 0]) * (self.p3[:, 1] - self.p1[:, 1]) -
                                  (self.p3[:, 0] - self.p1[:, 0]) * (self.p2[:, 1] - self.p1[:, 1]))

    def assemble_system(self):
        # Compute gradients of the basis functions (vectorized)
        # b_all and c_all have shape (ntri, 3)
        b_all = np.column_stack((self.p2[:, 1] - self.p3[:, 1],
                                self.p3[:, 1] - self.p1[:, 1],
                                self.p1[:, 1] - self.p2[:, 1]))
        c_all = np.column_stack((self.p3[:, 0] - self.p2[:, 0],
                                self.p1[:, 0] - self.p3[:, 0],
                                self.p2[:, 0] - self.p1[:, 0]))

        # Lists to collect the global indices and corresponding matrix entries
        self.matrix_row_list = []
        self.matrix_col_list = []
        self.matrix_data_list = []

        # Loop over local element matrix indices (9 iterations total)
        for i in range(3):
            for j in range(3):
                # For all triangles at once:
                # Compute Ke[i,j] = (b_i*b_j + c_i*c_j) / (4*area)
                vals = (b_all[:, i] * b_all[:, j] + c_all[:, i] * c_all[:, j]) / (4 * self.areas)
                # include materials:
                # vals *= self.material
                # Global indices for the contribution:
                self.matrix_row_list.append(self.elem[:, i])
                self.matrix_col_list.append(self.elem[:, j])
                self.matrix_data_list.append(vals)

        # Concatenate all contributions to form vectors of row indices, column indices, and values.
        self.matrix_rows = np.concatenate(self.matrix_row_list)
        self.matrix_cols = np.concatenate(self.matrix_col_list)

        # Assemble the global stiffness matrix (in COO format, then convert to CSR)
        # Assemble the load vector.
        self.change_material(self.material)

    def assemble_system_elementwise(self):
        """Assemble the global stiffness matrix and RHS vector elementwise.
        Only for testing purposes."""
        A = lil_matrix((self.n_node, self.n_node))
        self.system_rhs = np.zeros(self.n_node)

        for i in range(self.n_elem):
            x = self.node[self.elem[i, :], :]

            # Compute gradient matrix B
            B_ = np.array([
                [x[1, 1] - x[2, 1], x[2, 1] - x[0, 1], x[0, 1] - x[1, 1]],
                [x[2, 0] - x[1, 0], x[0, 0] - x[2, 0], x[1, 0] - x[0, 0]]
            ])

            # Local stiffness matrix
            A_local = self.material[i] * (B_.T @ B_) / (4 * self.areas[i])

            # Assemble global stiffness matrix
            for m in range(3):
                for n in range(3):
                    A[self.elem[i, m], self.elem[i, n]] += A_local[m, n]

            # Local RHS
            b_local = np.ones(3) / 3 * self.source[i] * self.areas[i]

            # Assemble global RHS vector
            self.system_rhs[self.elem[i, :]] += b_local

        self.system_matrix = A.tocsr()

    def change_material(self, material):
        """Change the material values and reassemble the system matrix."""
        # if material is a scalar, set to constant material value
        if isinstance(material, int):
            self.material = np.full(self.n_elem, material)
        # if material is a list, set to corresponding material values
        elif isinstance(material, list):
            self.material = np.array(material)
        # if material is a numpy array, set to corresponding material values
        elif isinstance(material, np.ndarray):
            self.material = material
        # if material is a function, compute material values for each element center
        elif callable(material):
            self.material = material(self.centers)
        # if material is None, set to constant 1 (default)
        else:
            self.material = np.ones(self.n_elem)

        self.matrix_data_list_with_material = [d*self.material for d in self.matrix_data_list]
        self.matrix_data = np.concatenate(self.matrix_data_list_with_material)

        # Assemble the global stiffness matrix (in COO format, then convert to CSR)
        self.system_matrix = coo_matrix((self.matrix_data, (self.matrix_rows, self.matrix_cols)), shape=(self.n_node, self.n_node)).tocsr()
        # For each triangle the load vector contribution is (area/3) for each vertex.
        self.system_rhs = np.bincount(self.elem.ravel(), weights=np.repeat(self.areas*self.source / 3, 3), minlength=self.n_node)

    def _apply_boundary_conditions(self):

        # apply Dirichlet boundary conditions
        self.solution[self.nodes_Dirichlet] = self.values_Dirichlet
        self.system_rhs = self.system_rhs - self.system_matrix @ self.solution  # Modify RHS for Dirichlet conditions

        # apply Neumann boundary conditions
        for i in range(len(self.values_Neumann)):
            x = self.node[self.edges_Neumann[i, :], :]
            self.system_rhs[self.edges_Neumann[i, :]] += np.linalg.norm(x[0, :] - x[1, :]) * self.values_Neumann[i] / 2

    def solve(self):
        # initialize solution vector
        self.solution = np.zeros(self.n_node)
        self._apply_boundary_conditions()

        # non-Dirichlet indices
        # self.free_nodes = np.where(~self.nodes_Dirichlet)[0]
        self.free_nodes = ~self.nodes_Dirichlet

        # solve the linear system
        self.solution[self.free_nodes] = spsolve(self.system_matrix[self.free_nodes, :][:, self.free_nodes], self.system_rhs[self.free_nodes])

    def get_state_function_observations(self, measurement_points):
        """Get the state function at the given measurement points."""
        return self.solution[measurement_points]

    def plot_state_function(self):
        # visualization using tricontourf
        plt.figure(figsize=(8, 6))
        contour = plt.tricontourf(self.coords_x, self.coords_y, self.solution, triangles=self.elem, cmap='viridis', levels=50)
        plt.axis('equal')
        plt.colorbar(contour, label="Solution u")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Solution Visualization")
        plt.show()

    def plot_material(self):
        # visualization of material values in centres of triangles
        plt.figure(figsize=(8, 6))
        plt.tripcolor(self.coords_x, self.coords_y, self.elem, facecolors=self.material, cmap='viridis')
        plt.axis('equal')
        plt.colorbar(label="Material Value")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Material Visualization")

    def plot_mesh(self):
        # visualization of the mesh
        plt.figure(figsize=(8, 6))
        plt.triplot(self.coords_x, self.coords_y, self.elem, color='k', linewidth=0.5)
        plt.axis('equal')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Mesh Visualization")

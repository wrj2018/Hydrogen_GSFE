# HaoyangHe - 2025-11-20: ヽ(•‿•)ノ
# Integrated class for both T-site and O-site detection in BCC crystal structure

import numpy as np
from scipy.spatial import KDTree
import random
from typing import List, Tuple, Dict, Any, Optional, Union

class BCCHydrogenInserter:
    """Integrated class for identifying both tetrahedral (T) and octahedral (O) sites in BCC crystal and inserting hydrogen atoms."""
    
    def __init__(self, lattice_constant: float = 2.834, 
                 distance_tolerance: float = 0.01, 
                 angle_tolerance: float = 5.0,
                 max_neighbors: int = 14,
                 random_seed: int = 666,
                 layer_bias: List[int] = [0]):
        """
        Initialize the hydrogen inserter.
        
        Args:
            lattice_constant: BCC Fe lattice constant (Å)
            distance_tolerance: Distance tolerance for neighbor identification
            angle_tolerance: Angle tolerance for tetrahedral site identification (degrees)
            max_neighbors: Maximum number of neighbors to consider
            random_seed: Random seed for reproducibility
        """
        self.lattice_constant = lattice_constant
        self.distance_tolerance = distance_tolerance
        self.angle_tolerance = angle_tolerance
        self.max_neighbors = max_neighbors
        self.random_seed = random_seed
        
        # Pre-calculate BCC structure distances
        self.d1 = np.sqrt(3) / 2 * lattice_constant  # Nearest neighbor distance
        self.d2 = lattice_constant  # Second neighbor distance
        self.layer_gap = lattice_constant / np.sqrt(2)

        # fliter parameters
        self.min_distance = self.lattice_constant * np.sqrt(5) / 4
        self.layer_thickness = self.lattice_constant / np.sqrt(2)
        self.layer_bias = np.array(layer_bias) * self.layer_thickness
        
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    def read_lammps_data(self, filename: str) -> Dict[str, Any]:
        """Read LAMMPS data file and extract atom positions and box information."""
        with open(filename, 'r') as f:
            lines = f.readlines()
        num_atoms = int(lines[2].split()[0])
        # Parse box boundaries
        xlo, xhi = map(float, lines[5].split()[:2])
        ylo, yhi = map(float, lines[6].split()[:2])
        zlo, zhi = map(float, lines[7].split()[:2])
        
        box_size = np.array([xhi - xlo, yhi - ylo, zhi - zlo])
        box_bounds = [xlo, xhi, ylo, yhi, zlo, zhi]
        
        # Find atoms section
        atoms_start = None
        for i, line in enumerate(lines):
            if line.strip() == "Atoms # atomic":
                atoms_start = i + 2
                break
        
        # Parse atom data
        atoms = []
        atom_ids = []
        for i in range(atoms_start, len(lines)):
            if not lines[i].strip():
                break
            parts = lines[i].split()
            if len(parts) < 5 or parts[1] != '2':
                break
            atom_id = int(parts[0])
            atom_type = int(parts[1])
            x, y, z = map(float, parts[2:5])
            atoms.append([x, y, z])
            atom_ids.append(atom_id)
        
        return {
            'atoms': np.array(atoms),
            'atom_ids': atom_ids,
            'box_size': box_size,
            'box_bounds': box_bounds,
            'lines': lines,
            'atom_count': num_atoms
        }
    
    def pbc_distance(self, pos1: np.ndarray, pos2: np.ndarray, box_size: np.ndarray) -> float:
        """Calculate minimum image distance between two positions with PBC."""
        diff = pos1 - pos2
        diff -= box_size * np.round(diff / box_size)
        return np.linalg.norm(diff)
    
    def pbc_vector(self, pos1: np.ndarray, pos2: np.ndarray, box_size: np.ndarray) -> np.ndarray:
        """Calculate minimum image vector from pos1 to pos2 with PBC."""
        diff = pos2 - pos1
        diff -= box_size * np.round(diff / box_size)
        return diff
    
    def pbc_midpoint(self, pos1: np.ndarray, pos2: np.ndarray, 
                    box_size: np.ndarray, box_bounds: List[float]) -> np.ndarray:
        """Calculate midpoint between two positions with PBC correction."""
        xlo, xhi, ylo, yhi, zlo, zhi = box_bounds
        
        diff = self.pbc_vector(pos1, pos2, box_size)
        midpoint = pos1 + diff / 2.0
        
        # Apply PBC to ensure midpoint is within box
        for i, (lo, hi, size) in enumerate([(xlo, xhi, box_size[0]), 
                                           (ylo, yhi, box_size[1]), 
                                           (zlo, zhi, box_size[2])]):
            midpoint[i] = lo + (midpoint[i] - lo) % size
            if midpoint[i] >= hi - 1e-10:
                midpoint[i] = hi - 1e-10
            elif midpoint[i] < lo:
                midpoint[i] = lo
        
        return midpoint
    
    def apply_pbc_to_point(self, point: np.ndarray, box_bounds: List[float], 
                          box_size: np.ndarray) -> np.ndarray:
        """Apply periodic boundary conditions to a point."""
        xlo, xhi, ylo, yhi, zlo, zhi = box_bounds
        result = point.copy()
        
        for i, (lo, hi, size) in enumerate([(xlo, xhi, box_size[0]), 
                                           (ylo, yhi, box_size[1]), 
                                           (zlo, zhi, box_size[2])]):
            result[i] = lo + (result[i] - lo) % size
            
            if result[i] >= hi - 1e-10:
                result[i] = hi - 1e-10
            elif result[i] < lo:
                result[i] = lo
        
        return result
    
    def find_tetrahedral_sites_bcc(self, atoms: np.ndarray, box_size: np.ndarray, 
                                   box_bounds: List[float], target_axis: int = 1) -> Tuple[List[np.ndarray], float, int]:
        """Find tetrahedral void sites in BCC crystal structure (Optimized Neighbor Search)."""
        xlo, xhi, ylo, yhi, zlo, zhi = box_bounds
    
        if target_axis == 0:  # x
            axis_mid = (xlo + xhi) / 2
            axis_bounds = [xlo, xhi]
        elif target_axis == 1:  # y
            axis_mid = (ylo + yhi - self.lattice_constant / np.sqrt(2)) / 2 + self.layer_bias
            # different from O site, we need to pick the mid of y-layer atom instead of box.
            axis_bounds = [ylo, yhi]
        else:  # z
            axis_mid = (zlo + zhi) / 2
            axis_bounds = [zlo, zhi]

        half_thickness = self.lattice_constant / np.sqrt(2) / 2 + 0.01
        target_layer = (axis_mid - half_thickness, axis_mid + half_thickness)
        
        # Find atoms in target layer
        target_atoms_indices = []
        for i, atom_pos in enumerate(atoms):
            if np.min(target_layer[0]) <= atom_pos[target_axis] <= np.max(target_layer[1]):
                target_atoms_indices.append(i)
        
        print(f"Number of atoms in target layer: {len(target_atoms_indices)}")
        
        if len(target_atoms_indices) == 0:
            print("ERROR: No atoms found in the target layer.")
            return [], axis_mid, 0
        
        if len(atoms) == 0:
             return [], 0, 0

        tree = KDTree(atoms, boxsize=box_size)

        search_radius = self.d2 + self.distance_tolerance * 2 
        
        tetra_sites = []
        used_sites = set()
        
        for i in range(len(atoms)):
            center_atom = atoms[i]

            distances, neighbor_indices = tree.query(
                center_atom, 
                k=self.max_neighbors + 1, 
                distance_upper_bound=search_radius
            )

            nearest_neighbors = [] # Indices for d1 (NN)
            second_neighbors = []  # Indices for d2 (2NN)

            for idx, dist in zip(neighbor_indices[1:], distances[1:]):

                if dist > search_radius + 1e-6:
                    break 
                if abs(dist - self.d1) < self.distance_tolerance:
                    nearest_neighbors.append(idx)
                elif abs(dist - self.d2) < self.distance_tolerance:
                    second_neighbors.append(idx)

            midpoints_second_nn = []
            for neighbor_idx in second_neighbors:
                midpoint = self.pbc_midpoint(center_atom, atoms[neighbor_idx], box_size, box_bounds)
                midpoints_second_nn.append(midpoint)
            
            for midpoint_second in midpoints_second_nn:
                for k in range(len(nearest_neighbors)):
                    for l in range(k + 1, len(nearest_neighbors)):
                        idx_k = nearest_neighbors[k]
                        idx_l = nearest_neighbors[l]
                        
                        vec1 = self.pbc_vector(midpoint_second, atoms[idx_k], box_size)
                        vec2 = self.pbc_vector(midpoint_second, atoms[idx_l], box_size)
                        
                        norm1 = np.linalg.norm(vec1)
                        norm2 = np.linalg.norm(vec2)
                        
                        if norm1 > 1e-8 and norm2 > 1e-8:
                            cos_angle = np.dot(vec1, vec2) / (norm1 * norm2)
                            
                            if abs(cos_angle) < 1e-8: 

                                midpoint_nearest = self.pbc_midpoint(atoms[idx_k], 
                                                                     atoms[idx_l], 
                                                                     box_size, box_bounds)
                                
                                tetra_site = self.pbc_midpoint(midpoint_second, midpoint_nearest, 
                                                               box_size, box_bounds)
                                
                                site_pbc = self.apply_pbc_to_point(tetra_site, box_bounds, box_size)
                                
                                # Avoid duplicate sites
                                pos_key = tuple(round(coord, 8) for coord in site_pbc)
                                if pos_key not in used_sites:
                                    tetra_sites.append(site_pbc)
                                    used_sites.add(pos_key)

        return tetra_sites, axis_mid, len(target_atoms_indices)
    
    def find_octahedral_sites_full_system(self, atoms: np.ndarray, box_size: np.ndarray, 
                                         box_bounds: List[float]) -> List[np.ndarray]:
        """
        Find all octahedral sites in the entire system.
        
        This function identifies octahedral sites by finding midpoints between
        atom pairs separated by the lattice constant distance.
        """
        o_sites = []
        used_pairs = set()

        xlo, xhi, ylo, yhi, zlo, zhi = box_bounds

        tree = KDTree(atoms, boxsize=box_size)

        for i, atom_pos in enumerate(atoms):
            # Find neighbors within lattice constant + tolerance
            indices = tree.query_ball_point(atom_pos, self.lattice_constant + self.distance_tolerance)

            for j in indices:
                if j == i:
                    continue

                # Calculate distance with PBC
                dist = self.pbc_distance(atom_pos, atoms[j], box_size)

                # Check if distance matches lattice constant
                if abs(dist - self.lattice_constant) < self.distance_tolerance:
                    # Calculate midpoint
                    midpoint = self.pbc_midpoint(atom_pos, atoms[j], box_size, box_bounds)

                    # Ensure midpoint is within box bounds
                    if (xlo <= midpoint[0] <= xhi and
                        ylo <= midpoint[1] <= yhi and
                        zlo <= midpoint[2] <= zhi):

                        # Use rounded coordinates as unique key to avoid duplicates
                        pos_key = tuple(round(coord, 3) for coord in midpoint)

                        if pos_key not in used_pairs:
                            o_sites.append(midpoint)
                            used_pairs.add(pos_key)

        return o_sites
    
    def filter_sites_by_y_layer(self, sites: List[np.ndarray], box_bounds: List[float]) -> List[np.ndarray]:

        xlo, xhi, ylo, yhi, zlo, zhi = box_bounds

        y_mid = (ylo + yhi - self.lattice_constant / np.sqrt(2)) / 2 + self.layer_bias

        filtered_sites = []

        # Ensure insert_layers is always defined as a list
        if isinstance(y_mid, (list, np.ndarray)):
            insert_layers = list(y_mid)
        else:
            insert_layers = [y_mid]
    
        for site in sites:
            y_coord = site[1]
                
            in_range = any(layer_y - self.layer_thickness / 2 <= y_coord <= layer_y + self.layer_thickness / 2 
                        for layer_y in insert_layers)

            divisible = any(abs(y_coord - layer_y - self.layer_thickness / 2) < 1e-4 or 
                        abs(y_coord - layer_y + self.layer_thickness / 2) < 1e-4
                        for layer_y in insert_layers)
                
            if not in_range or divisible:
                continue
                
            filtered_sites.append(site)         

        return filtered_sites
    
    def filter_sites_by_distance(self, sites: List[np.ndarray], atoms: np.ndarray, 
                                box_size: np.ndarray) -> List[np.ndarray]:
        if not sites:
            return []
        
        tree = KDTree(atoms, boxsize=box_size)
        filtered_sites = []
        
        for site in sites:
            dist_to_nearest, _ = tree.query(site, k=1)
            if abs(dist_to_nearest - self.min_distance) <= 1e-3:
                filtered_sites.append(site)
        
        return filtered_sites
    
    def insert_hydrogen_atoms(self, data: Dict[str, Any], sites: List[np.ndarray], 
                             concentration: float = 1.0, 
                             site_type: str = "H",
                             atom_type: int = 1) -> Tuple[List[str], int]:
        """
        Insert hydrogen atoms into interstitial sites.
        
        Args:
            data: Dictionary containing atom data and file lines
            sites: List of interstitial site coordinates
            concentration: Fraction of sites to fill with hydrogen (0.0 to 1.0)
            site_type: Type of site ("H" for hydrogen, can be used for labeling)
            atom_type: Atom type ID for the inserted atoms
            
        Returns:
            Tuple of (updated_lines, num_atoms_inserted)
        """
        original_lines = data['lines']
        atom_ids = data['atom_ids']
        original_atom_count = data['atom_count']
        
        # Determine number of atoms to insert
        num_insert_atoms = int(data['atom_count'] * concentration)
        num_insert_atoms = min(num_insert_atoms, len(sites))

        if num_insert_atoms == 0:
            print("Warning: No atoms to insert based on the given concentration.")
            return original_lines, 0

        # Randomly select sites for insertion
        selected_sites = random.sample(sites, num_insert_atoms)

        print(f"Atoms to insert: {num_insert_atoms}")
        print(f"Original atoms: {original_atom_count}")
        print(f"Total atoms: {original_atom_count + num_insert_atoms}")

        # Update LAMMPS data file
        new_lines = []
        header_updated = False
        
        for line in original_lines:
            if "atoms" in line and not line.startswith("Atoms") and not header_updated:
                num_new_atoms = original_atom_count + num_insert_atoms
                new_lines.append(f"{num_new_atoms} atoms\n")
                header_updated = True
            elif "atom types" in line:
                parts = line.split()
                if len(parts) >= 1 and parts[0].isdigit():
                    num_old_types = int(parts[0])
                    new_types = max(atom_type + 1, num_old_types)
                    new_lines.append(f"{new_types} atom types\n")
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        
        # Find atoms section
        atoms_start = -1
        for i, line in enumerate(new_lines):
            if line.strip() == "Atoms # atomic":
                atoms_start = i
                break
        
        if atoms_start == -1:
            print("ERROR: Atoms section not found.")
            return original_lines, 0
        
        # Find end of atoms section
        atoms_end = -1
        for i in range(atoms_start + 2, len(new_lines)):
            if not new_lines[i].strip():
                atoms_end = i
                break
            if new_lines[i].strip() and not new_lines[i].split()[0].isdigit():
                atoms_end = i
                break
        
        if atoms_end == -1:
            atoms_end = len(new_lines)
        
        # Add new atoms
        new_atom_lines = []
        for k, site in enumerate(selected_sites):
            atom_id = original_atom_count + k + 1
            new_atom_lines.append(f"{atom_id} {atom_type} {site[0]:.12f} {site[1]:.12f} {site[2]:.12f}\n")
        
        new_lines[atoms_end:atoms_end] = new_atom_lines
        
        return new_lines, num_insert_atoms
    
    def process_tetrahedral_sites(self, input_file: str, output_file: str, 
                                 h_concentration: float = 1.0, 
                                 target_axis: int = 1) -> Dict[str, Any]:
        
        data = self.read_lammps_data(input_file)

        print("\nFinding tetrahedral voids in BCC crystal...")
        T_tetra_sites, _, _ = self.find_tetrahedral_sites_bcc(
            data['atoms'], data['box_size'], data['box_bounds'], target_axis=target_axis
        )
        
        if len(T_tetra_sites) == 0:
            print("No tetrahedral voids found. Exiting.")
            return {'success': False, 'error': 'No tetrahedral sites found'}
        
        T_tetra_sites = self.filter_sites_by_distance(T_tetra_sites, data['atoms'], 
                                                       data['box_size'])
        
        T_tetra_sites = self.filter_sites_by_y_layer(T_tetra_sites, data['box_bounds'])
        
        new_lines, num_inserted = self.insert_hydrogen_atoms(
            data, T_tetra_sites, concentration=h_concentration,
            site_type="H", atom_type=1
        )
        
        # Write output file
        with open(output_file, 'w') as f:
            f.writelines(new_lines)
        
        # Print summary
        print(f"\n=== T-site Results Summary ===")
        print(f"Original Fe atoms: {data['atom_count']}")
        print(f"Tetrahedral sites found: {len(T_tetra_sites)}")
        print(f"H atoms inserted: {num_inserted}")
        print(f"Concentration: {h_concentration * 100:.3f}%")
        print(f"Output file: {output_file}")
        
        return {
            'success': True,
            'site_type': 'tetrahedral',
            'original_atoms': data['atom_count'],
            'sites_found': len(T_tetra_sites),
            'atoms_inserted': num_inserted,
            'concentration': h_concentration
        }
    
    def process_octahedral_sites(self, input_file: str, output_file: str, 
                                h_concentration: float = 1.0) -> Dict[str, Any]:
        """Process octahedral sites for hydrogen insertion."""
        data = self.read_lammps_data(input_file)
        o_sites_all = self.find_octahedral_sites_full_system(data['atoms'], data['box_size'], data['box_bounds'])

        if len(o_sites_all) == 0:
            print("No O-sites found. Exiting.")
            return {'success': False, 'error': 'No O-sites found'}
        
        if self.layer_thickness is not None:
            o_sites_filtered = self.filter_sites_by_y_layer(o_sites_all, data['box_bounds'])
        else:
            o_sites_filtered = o_sites_all
            print("No layer filtering applied, using all O-sites")

        # Step 4: Insert hydrogen atoms
        new_lines, num_inserted = self.insert_hydrogen_atoms(
            data, o_sites_filtered, concentration=h_concentration,
            site_type="H", atom_type=1
        )

        # Write output file
        with open(output_file, "w") as f:
            f.writelines(new_lines)

        # Print summary
        print(f"\n=== O-site Results Summary ===")
        print(f"Original Fe atoms: {data['atom_count']}")
        print(f"O-sites found: {len(o_sites_all)}")
        print(f"O-sites after filtering: {len(o_sites_filtered)}")
        print(f"H atoms inserted: {num_inserted}")
        print(f"Concentration: {h_concentration * 100:.1f}%")
        print(f"Output file: {output_file}")

        return {
            'success': True,
            'site_type': 'octahedral',
            'original_atoms': data['atom_count'],
            'sites_found': len(o_sites_all),
            'sites_filtered': len(o_sites_filtered),
            'atoms_inserted': num_inserted,
            'concentration': h_concentration
        }


def main():
    """Example usage of the integrated BCCHydrogenInserter class."""
    inserter = BCCHydrogenInserter(
        lattice_constant=2.834,
        distance_tolerance=0.1,
        angle_tolerance=2.0,
        max_neighbors=14,
        random_seed=666,
        layer_bias=0
    )

    # Example 1: Process tetrahedral sites
    print("=== Processing Tetrahedral Sites ===")
    result_t = inserter.process_tetrahedral_sites(
        input_file="100_0_0_0_0/100_0_0_0_0.lmp",
        output_file="100_0_0_0_0/100_0_0_0_0_T.lmp",
        h_concentration=1,
        target_axis=1
    )
    
    # Example 2: Process octahedral sites  
    print("\n=== Processing Octahedral Sites ===")
    result_o = inserter.process_octahedral_sites(
        input_file="100_0_0_0_0/100_0_0_0_0.lmp",
        output_file="100_0_0_0_0/100_0_0_0_0_O.lmp",
        h_concentration=1
    )

if __name__ == "__main__":
    main() 

print(__name__)
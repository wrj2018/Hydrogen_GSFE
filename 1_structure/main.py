from VoidSiteBCC import BCCHydrogenInserter
import os

# Create output directory if it doesn't exist
output_dir = "layer_bias"
os.makedirs(output_dir, exist_ok=True)

base_file = "single.lmp"
all_layers = [-1,0,1]  # Process layers -1, 0, and 1 sequentially
target_concentration_per_layer = 0.0005  # 0.15%
num_replicas = 5  # 5 different distributions

# Create 5 replicas with different random seeds
for replica_idx, random_seed in enumerate(range(1, num_replicas + 1)):
    current_file = base_file
    output_file = f"{output_dir}/0.15%_uniform_layer_{replica_idx + 1}.lmp"
    
    # Process each layer sequentially to ensure 0.15% concentration in each layer
    for layer in all_layers:
        # Create inserter for specific layer with current random seed
        print(random_seed + replica_idx * 100 + layer + 222)
        inserter = BCCHydrogenInserter(
            lattice_constant=2.834,
            distance_tolerance=0.1,
            angle_tolerance=0.1,
            max_neighbors=14,
            random_seed=random_seed + replica_idx * 100 + layer + 222,
            layer_bias=[layer]
        )
        
        # Process single layer with 0.15% concentration
        temp_output = f"{output_dir}/temp_{replica_idx + 1}_{layer}.lmp"
        inserter.process_tetrahedral_sites(
            input_file=current_file,
            output_file=temp_output,
            h_concentration=target_concentration_per_layer
        )
        # Update current file for next layer insertion
        current_file = temp_output
    
    # Rename final file
    if os.path.exists(current_file):
        os.rename(current_file, output_file)
    
    # Clean up any extra temp files if needed
    print(f"\nCompleted replica {replica_idx + 1} with random seed {random_seed}")
    print(f"Final output: {output_file}")




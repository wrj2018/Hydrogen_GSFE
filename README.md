# Hydrogen_GSFE

Generalized Stacking Fault Energy (GSFE) calculations for hydrogen-containing BCC $\alpha$-Fe using atomistic simulations. Deep-learning Fe-H interatomic potential based on DeePMD-kit framework.

## Environment Requirements

### Software
- **Python 3.11+**
- **LAMMPS** (compiled with DeePMD-kit interface supported)
- **DeePMD-kit** (v2.7+)
- **Atomsk** (v0.13.1+, for structure generation)

### Python Dependencies
- `numpy`
- `pandas`

### NNP Potential

The Fe-H deep potential file `FeH.pb` is located in `0_NNP/`.  
All LAMMPS input scripts reference it via relative path `../../../0_NNP/FeH.pb`.

## Usage

### 1. Generate single crystal structure，insert hydrogen atoms
```bash
cd 1_structure
python generate_RandomSFE_Input_Atomsk.py
python main.py
```

### 2. Run GSFE calculations
Modify `submit.sh` to set your HPC account/partition, then:
```bash
cd 2_script/110-concentrate-0.05-0.15%/<seed_dir>
sbatch submit.sh
```

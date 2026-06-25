import os
import re


def generate_single_bcc(lattice_constant=2.834):
    # powerShell_Input = "atomsk --create bcc {} Fe orient [1-11] [121] [-101] -duplicate 41 17 25 single.lmp".format(lattice_constant)
    powerShell_Input = "atomsk --create bcc {} Fe orient [1-11] [-101] [121] -duplicate 41 30 14 single.lmp".format(lattice_constant)
    os.system("powershell.exe {}".format(powerShell_Input))


def convert_to_fe_h_format(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    new_lines = []
    skip_masses = False
    in_atoms = False

    for line in lines:
        if line.strip().startswith("Masses") and not in_atoms:
            skip_masses = True
            continue
        if "Atoms" in line and not in_atoms:
            skip_masses = False
            in_atoms = True
            new_lines.append("Masses\n\n")
            new_lines.append("1 1.008 #H\n")
            new_lines.append("2 55.845 #Fe\n\n")
            new_lines.append(line)
            continue

        if skip_masses:
            continue

        if in_atoms:
            if line.strip() == "" or line.strip().startswith("#"):
                new_lines.append(line)
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                parts[1] = "2"
                new_lines.append(" ".join(parts) + "\n")
            else:
                new_lines.append(line)
        else:
            if "1  atom types" in line:
                line = line.replace("1  atom types", "2  atom types")
            new_lines.append(line)

    with open(output_file, 'w') as f:
        f.writelines(new_lines)


if __name__ == "__main__":
    generate_single_bcc()
    convert_to_fe_h_format("single.lmp", "single.lmp")
    print("Done: single.lmp (pure Fe BCC, 2 atom types)")

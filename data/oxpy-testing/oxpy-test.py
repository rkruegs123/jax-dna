import oxpy
from oxDNA_analysis_tools.UTILS.RyeReader import describe
from oxDNA_analysis_tools.output_bonds import output_bonds

input_path = "/home/ryan/Documents/Harvard/research/brenner/jaxmd-oxdna/data/polyA_10bp/input_dummy"
top_path = "/home/ryan/Documents/Harvard/research/brenner/jaxmd-oxdna/data/polyA_10bp/generated.top"
traj_path = "/home/ryan/Documents/Harvard/research/brenner/jaxmd-oxdna/data/polyA_10bp/generated.dat"

if __name__ == "__main__":
    top_info, traj_info = describe(top_path, traj_path)
    output_bonds(traj_info, top_info, input_path, visualize=False)

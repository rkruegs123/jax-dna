from test import test_subterms

def run():
    traj_path = "../oxDNA/examples/HAIRPIN/initial.conf"
    top_path = "../oxDNA/examples/HAIRPIN/initial.top"
    input_path = "../oxDNA/examples/HAIRPIN/input"
    print(test_subterms.run(top_path, traj_path, input_path))

if __name__ == "__main__":
    run()

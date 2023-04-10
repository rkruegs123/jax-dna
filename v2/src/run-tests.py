from pathlib import Path
import pdb

from test import test_subterms, test_propeller, test_pitch


def run():
    # Test subterms tests

    """
    basedir = Path("v2/data/test-data/unbound-strands-overlap")
    traj_path = basedir / "output.dat"
    top_path = basedir / "generated.top"
    input_path = basedir / "input"
    test_subterms.run(top_path, traj_path, input_path)

    pdb.set_trace()
    """


    for use_neighbors in [True, False]:
        basedir = Path("v2/data/test-data/simple-coax")
        traj_path = basedir / "output.dat"
        top_path = basedir / "generated.top"
        input_path = basedir / "input"
        test_subterms.run(top_path, traj_path, input_path, use_neighbors=use_neighbors)

        basedir = Path("v2/data/test-data/simple-helix")
        traj_path = basedir / "output.dat"
        top_path = basedir / "generated.top"
        input_path = basedir / "input"
        test_subterms.run(top_path, traj_path, input_path, use_neighbors=use_neighbors)

    # Test propeller twist
    test_propeller.run()

    # Test pitch
    test_pitch.run()

if __name__ == "__main__":
    run()

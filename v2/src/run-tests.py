from pathlib import Path

from test import test_subterms, test_propeller, test_pitch


def run():
    # Test subterms tests
    basedir = Path("v2/data/test-data/simple-coax")
    traj_path = basedir / "output.dat"
    top_path = basedir / "generated.top"
    input_path = basedir / "input"
    test_subterms.run(top_path, traj_path, input_path)

    basedir = Path("v2/data/test-data/simple-helix")
    traj_path = basedir / "output.dat"
    top_path = basedir / "generated.top"
    input_path = basedir / "input"
    test_subterms.run(top_path, traj_path, input_path)

    # Test propeller twist
    test_propeller.run()

    # Test pitch
    test_pitch.run()

if __name__ == "__main__":
    run()

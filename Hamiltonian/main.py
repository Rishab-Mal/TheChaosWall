import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Hamiltonian.train import train_hnn

if __name__ == "__main__":
    hnn = train_hnn()
    print("HNN training complete.")

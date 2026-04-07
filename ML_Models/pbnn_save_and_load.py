
import os
import numpy as np
import pickle
import neurobayes as nb


def save_partial_bnn(model, folder):
    os.makedirs(folder, exist_ok=True)
    # 1. Save deterministic weights
    det_w = model._model.deterministic_weights
    np.save(os.path.join(folder, "deterministic_weights.npy"),
            det_w, allow_pickle=True)
    print("Saved deterministic weights.")
    # 2. Save posterior MCMC samples
    samples = model._model.mcmc.get_samples(group_by_chain=False)
    with open(os.path.join(folder, "posterior_samples.pkl"), "wb") as f:
        pickle.dump(samples, f)
    print("Saved posterior samples.")

def load_partial_bnn(architecture,
                     probabilistic_settings=None,
                     folder="partial_bnn_saved"):
    # Load deterministic weights
    det_w = np.load(os.path.join(folder, "deterministic_weights.npy"),
                    allow_pickle=True).item()
    model = nb.PartialBNN(
        architecture,
        deterministic_weights=det_w,
        **(probabilistic_settings or {})
    )
    # Load posterior samples
    with open(os.path.join(folder, "posterior_samples.pkl"), "rb") as f:
        samples = pickle.load(f)
    # Attach a dummy MCMC object so .predict() works
    class DummyMCMC:
        def __init__(self, samples):
            self._samples = samples
        def get_samples(self, group_by_chain=False):
            return self._samples
    model._model.mcmc = DummyMCMC(samples)
    print(f"Loaded Partial BNN from folder: {folder}")
    return model
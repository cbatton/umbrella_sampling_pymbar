# Reweighting via PyMBAR with bootstrapping

The script contained in this folder implements reweighting via PyMBAR.
Install the needed Python packages and it should be ready to go.
This script started as an adaptation of one of the example scripts provided with PyMBAR, the one in which a 1D PMF is computed from an umbrella sampling simulation.
This was then extended to handle reweighting with respect to an external field, and then on top of that bootstrapping is use to develop a generally better error estimate than the PyMBAR defaults along with being an easier way to handle processing larger amounts of data than just doing one PyMBAR run.

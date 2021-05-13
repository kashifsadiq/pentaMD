# Pentasaccharide MD Simulation Repository

########################################################################################################
 Correspondence: S. Kashif Sadiq (kashif.sadiq@embl.de)
 Affiliation: 1. Heidelberg Institute for Theoretical Studies, HITS gGmbH
 2. European Moelcular Biology Laboratory
########################################################################################################
Last revised: 27/02/2021
########################################################################################################

This repository contains MD simulation input data, post-production analysis data and analysis scripts for the manuscript:

Balogh, Gabor; Gyöngyösi, Tamás; Timári, István; Herczeg, Mihály; Borbás, Anikó; Sadiq, S. Kashif; Fehér, Krisztina; Kövér, Katalin, Conformational Analysis of Heparin-Analogue Pentasaccharides by Nuclear Magnetic Resonance Spectroscopy and Molecular Dynamics Simulations (2021), Journal of Chemical Information and Modelling, Accepted.

MD Input Data

############################

MD input data is contained within the tarfile: pentaMD_input.tar (8.2 Mb)
To untar the file: tar xvf pentaMD_input.tar
Follow the instuctions in the README file contained in the tar file to navigate the input files. 

Post-production output data

############################

Post-production output data consists of the Cremer-Pople parameters (theta, phi) for each of the five rings in each pentasaccharide as well as the inter-glycosidic linker dihedral angles (phi_gl and psi_gl) for every recorded timestep in each replica and is available in the following files and corresponding file structure in this repository:

analysis/data/CP_$SYS_$REP.dat (18 Mb each)

analysis/data/GL_$SYS_$REP.dat (16 Mb each)


$SYS denotes the analog number from 1 to 3 

$REP denotes the replica number from 1 to 10

CP denotes the Cremer-Pople data and the columns successively report the theta and phi values for each of the rings in order from the D to H.

GL denotes the glycosidic linker data and the columns successively report the phi and psi values for each of the linkersin order from D/E to G/H.

Each file contains 200,000 rows ordered by the timepoint from star to end in each simulation with intervals of 10 ps.


Analysis scripts
############################

Analysis scripts are found in the folder 'analysis'. These are:

pentaMD_analysis.ipynb : A Python3 Jupyter Notebook that can be followed to reproduce the analysis in the manuscript, from the above post-production output data

and two python modules-
pentamd.py : contains functions for performing conformational and RMSD analyses and associated plotting functions 
msmanalysis.py : contains functions for performing Markov state model analysis based on the pyEMMMA software

These modules are imported from within the Jupyter Notebook


########################################################################################################


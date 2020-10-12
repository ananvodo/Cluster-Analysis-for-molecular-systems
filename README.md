# Geometry Description parameters using Cluster Analysis single linkage algorithm for MD simulations in GMX
(for python 3.x)

This is a program to determine clusters created by molecules in multiple molecular dynamics system. It calculates the Gyration Tensor to calculate geometry description parameters: Shape Description, Asphericity, Acylindricity, Radius of Gyration, and cluster mass content.
The calculation of the geometry description parameters are according to the publication: xxxx

This program is compatible with GROMACS itp format for molecular topologies.
It is compatible with argparse. For more information run: python cluster_analysis.py -h

The inputs are the following:
a) '-g' GRO file. It must comply with gromacs GRO format. If no -g is passed, the program will read all the gro files inside the workdir. This program aims to understand the evolution and geometry description parameters of clusters according to simulation time. Therefore, the GRO files must be consecutive in time. As example, sys1.gro, sys2.gro, sys3.gro, sys1000.gro; where sys1.gro is the frame of the simulation's initial time, and sys1000.gro is the last sim time frame.

b) '-m' is the itp file. It must comply with gromacs GRO format. If no -p is passed, the program will read martini_v2.2P.itp (as the default topology file when using martiniFF). This file is ONLY used to calculate mass% of the molecules in the cluster. In case other FF files want to be used, few changes must be done to the -m option in the code or eliminate the respected lines if %mass content is not needed.

c) '-p' is the itp file. It must comply with gromacs GRO format. If no -p is passed, the program will read Protein_A.itp (as the default itp output when using martiniFF). Any GMX itp file will be accepted.

d) '-i' are the output plots of all clusters' geometry description parameters in the systems. The x-axis is the time from beginning to end (according to the GRO files), and the y-axis is the geometry description parameters.

e) '-b' is the initial simulation time corresponding to the first GRO file (i.e., sys1.gro) in ns.

f) '-e' is the final simulation time corresponding to the last GRO file (i.e., sys1000.gro) in ns.


This program uses multiprocessing, meaning that multiple GRO files are analyzed at the same time.
Moreover, this program is intended to analyze water-oils systems and clusters created by bio-proteins. The default oils used are BENZ and DEC (according to martiniFF topology name), but they can be replaced by any other oil that wants to be used. Please consider that BENZ and DEC consist of only three beads. The number of beads must be according to the oil/molecule used. Please refer to lines 316 and 325-432.
NOTE: No itp file is required for the oil, only for the protein (in my case). If oil itp is required, then changes to the code must be done.





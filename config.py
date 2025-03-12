

#### CONFIG FOR 3D extruded IFW ####
DEBUG = True

path_to_directories = "/home/henrik/Documents/simulation_data/codeVerification/f1-ifw/eifw/"

directory_names = [
    "3d/please-work/physics/semiimplicit/dt1e-5/",
    "3d/please-work/physics/linearimplicit/dt1e-5/",
    "3d/please-work/physics/linearimplicit/dt5e-5/",
    "3d/please-work/physics/linearimplicit/dt1e-4/",
    "3d/please-work/physics/linearimplicit/dt2e-4/",
    "3d/please-work/physics/linearimplicit/dt5e-4/",
    "3d/please-work/physics/linearimplicit/dt1e-3/",
    # "3d/please-work/physics/substepping/dt2e-4/",
    # "3d/please-work/physics/substepping/dt1e-4/",
    # "3d/please-work/physics/substepping/dt5e-5/",
    # "3d/please-work/physics/substepping/dt1e-5/",
    ]

# Skip initial or final n points for each (individual/subdir!) force file
# Note that a small overlap removes the pressure kick after restarting
# We found that this improves PSDs
force_file_skip_start = 5
force_file_skip_end = 0


# case-specific definitions
ylabels = ["$C_d$", "$C_l$"]#, "$C_S$", "$C_L$"]
ynames = ["Drag", "Lift"]#, "Sheer", "Lift"]
customMetrics = ["F1-total", "F3-total"]#, "F2-total", "F3-total"]

# Characteristic velocity and lengths
freestream_velocity = 1.0 # non-dimensional velocity [m/s]
chord_length = 0.25 # mainplane chord [m]
ctu_len = chord_length / freestream_velocity # [s]
spanlen_npp = 0.05 # spanwise length for extruded IFW [m]
ref_area = ctu_len * spanlen_npp # reference area

# reference time step size (CFL ~= 1)
dtref = 1e-5



#### CONFIG FOR 2D cylinder ####
# path_to_directories = "/home/henrik/Documents/simulation_data/codeVerification/cylinder/2d/hx1/physics/"
#
# directory_names = [
#         "VCSImplicit/Extrapolated/IMEXOrder1/equal-order/dt1e-3/",
#         ]



#### CONFIG FOR minimal test ####
# path_to_directories = "/home/henrik/Documents/simulation_data/"
# directory_names = [
#     "test_files/"
# ]

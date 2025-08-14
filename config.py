
case = 'eifw'
DEBUG = True
use_cfl = False
use_iterations = True
scaling = True


#### CONFIG FOR 3D extruded IFW ####
if case == 'eifw':

    # Walk through directories and merge files
    log_file_glob_str = 'log.*'
    force_file_glob_strs = [
        # 'FWING_TOTAL_forces.fce',
        # 'LFW_fia_mp_forces.fce',
        # 'LFW_element_1_forces.fce',
        # 'LFW_element_2_forces.fce',
    ]
    history_file_glob_strs = [
        # 'mainplane_spanwise.his',
        # "mainplane-suction-midplane.his",
    ]

    path_to_directories = "/home/henrik/Documents/simulation_data/codeVerification/f1-ifw/eifw/"
    save_directory = "/home/henrik/Documents/simulation_data/cpc-figures/"

    directory_names = [
        # "quasi3d/james/farringdon_data/",
        # "3d/please-work/physics/semiimplicit/dt1e-5/",
        # "3d/please-work/physics/semiimplicit/dt1e-5-kinvis/",
        # "3d/please-work/physics/semiimplicit/dt1e-5/animation_280_285/",
        # "3d/please-work/physics/linearimplicit/dt1e-5/",
        # "3d/please-work/physics/linearimplicit/dt5e-5/",
        "3d/please-work/physics/linearimplicit/dt1e-4/",
        # "3d/please-work/physics/linearimplicit/dt2e-4/",
        # "3d/please-work/physics/linearimplicit/dt5e-4/",
        # "3d/please-work/physics/linearimplicit/dt1e-3/",
        # "3d/please-work/physics/substepping/dt1e-5/",
        # "3d/please-work/physics/substepping/dt5e-5/",
        # "3d/please-work/physics/substepping/dt1e-4/",
        # "3d/please-work/physics/substepping/dt2e-4/",
        # # "3d/please-work/physics/linearimplicit-strong/dt1e-4/",
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
    ref_velocity = 1.0 # non-dimensional velocity [m/s]
    chord_length = 0.25 # mainplane chord [m]
    ctu_len = chord_length / ref_velocity # [s]
    spanlen_npp = 0.05 # spanwise length for extruded IFW [m]
    ref_area = ctu_len * spanlen_npp # reference area

    # reference time step size (CFL ~= 1)
    dtref = 1e-5

    # divergence tolerance for detection
    divtol = 1e3

    # Boundary data naming:
    # b0 : main plane
    # b1 : 1st flap
    # b2 : 2nd flap
    boundary_names = [
        "b0",
        "b1",
        "b2",
    ]



#### CONFIG FOR 2D extruded IFW ####
elif case == '2difw':

    # Walk through directories and merge files
    log_file_glob_str = 'log*'
    force_file_glob_strs = [
        'FWING_TOTAL_forces.fce',
        'LFW_fia_mp_forces.fce',
        'LFW_element_1_forces.fce',
        'LFW_element_2_forces.fce',
    ]
    history_file_glob_strs = []

    path_to_directories = "/home/henrik/Documents/simulation_data/codeVerification/f1-ifw/eifw/2d/hx1/physics/"
    save_directory = "/home/henrik/Documents/simulation_data/gjp-figures/"

    directory_names = [
        # "semidt1e-6/",
        "pressure-weak-vs-strong/semidt5e-6/re2.2e5/dgsvv/ctu_35_45/",
        "pressure-weak-vs-strong/semidt5e-6/re2.2e5/gjp/ctu_35_45/",
        "pressure-weak-vs-strong/weakdt5e-6/re2.2e5/dgsvv/ctu_35_45/",
        "pressure-weak-vs-strong/weakdt5e-6/re2.2e5/gjp/ctu_35_45/",
        ]

    # Skip initial or final n points for each (individual/subdir!) force file
    # Note that a small overlap removes the pressure kick after restarting
    # We found that this improves PSDs
    force_file_skip_start = 5
    force_file_skip_end = 0


    # case-specific definitions
    ylabels = ["$C_d$", "$C_l$"]
    ynames = ["Drag", "Lift"]
    customMetrics = ["F1-total", "F2-total"]

    # Characteristic velocity and lengths
    ref_velocity = 1.0 # non-dimensional velocity [m/s]
    chord_length = 0.25 # mainplane chord [m]
    ctu_len = chord_length / ref_velocity # [s]
    ref_area = ctu_len # reference area

    # reference time step size (CFL ~= 1)
    dtref = 1e-6

    # divergence tolerance for detection
    divtol = 1e3

    # Boundary data naming:
    # b0 : main plane
    # b1 : 1st flap
    # b2 : 2nd flap
    boundary_names = [
        "b0",
        "b1",
        "b2",
    ]




#### CONFIG FOR 2D cylinder ####
elif case == '2dcyl':
    log_file_glob_str = 'log*'
    force_file_glob_strs = ['DragLift.fce']
    history_file_glob_strs = []
    path_to_directories = "/home/henrik/Documents/simulation_data/codeVerification/cylinder/2d/hx1/physics/"

    directory_names = [
            "VCSImplicit/Extrapolated/IMEXOrder1/equal-order/dt1e-3/",
            ]



#### CONFIG FOR minimal test ####
elif case == 'minimal':
    log_file_glob_str = 'log*'
    force_file_glob_strs = []
    history_file_glob_strs = []
    path_to_directories = "/home/henrik/Documents/simulation_data/"
    directory_names = [
        "test_files/"
    ]



# #### CONFIG FOR channel flow ####
elif case == 'channel':
    log_file_glob_str = 'log*'
    force_file_glob_strs = ['DragLift.fce']
    history_file_glob_strs = []
    path_to_directories = "/home/henrik/Documents/simulation_data/codeVerification/channel/physics/ret180/kmm-setup/"
    save_directory = "/home/henrik/Documents/simulation_data/codeVerification/channel/physics/"

    directory_names = [
        # "p8-dt1e-3/kick/ctu_01_378/",
        # "p8-dt1e-3/kick/ctu_360_/",
        # "p8-dt1e-3/kick-weak/ctu_01_50/",
        # "p8-dt1e-3/kick-dt2e-3/",
        # "p8-dt1e-3/kick-xxt/ctu_01_100/",
        # "p8-dt1e-3/flowrate/ctu_10_50/",
        # "p8-dt1e-3/flowrate/ctu_50_100/",
        "p8-dt1e-3/flowrate/ctu_100_260/",
        # "p8-dt1e-3/flowrate-weak/ctu_10_110/",
        # "p8-dt1e-3/flowrate-weak/ctu_110_310/",
        # "p8-dt1e-3/flowrate-re4200/ctu_10_300/",
        "p8-dt1e-3/flowrate/ctu_260__re4200/",
        "p8-dt1e-3/flowrate-weak/ctu_270__re4200/",
        "p8-dt1e-3/forcing/ctu_260__re4200/",
        ]


    # Skip initial or final n points for each (individual/subdir!) force file
    # Note that a small overlap removes the pressure kick after restarting
    # We found that this improves PSDs
    force_file_skip_start = 0
    force_file_skip_end = 0


    # case-specific definitions
    ylabels = ["$C_d$", "$C_l$", "$C_s$"]#, "$C_S$", "$C_L$"]
    ynames = ["Drag", "Lift", "Shear"]#, "Sheer", "Lift"]
    customMetrics = ["F1-total", "F2-total", "F3-total"]#, "F2-total", "F3-total"]

    # Characteristic velocity and lengths
    ref_velocity = 1.0 # non-dimensional velocity [m/s]
    chord_length = 1 # mainplane chord [m]
    ctu_len = chord_length / ref_velocity # [s]
    spanlen_npp = 1 # spanwise length for extruded IFW [m]
    ref_area = ctu_len * spanlen_npp # reference area

    # reference time step size (CFL ~= 1)
    dtref = 1e-3

    # Choose inidividual file names for averages
    # ctu_names = [
    #     "ctu_204_210",
    #     "ctu_204_242",
    #     "ctu_204_268",
    #     "ctu_204_287",
    # ]

    # Boundary data naming:
    # b0 : floor
    boundary_names = [
        "b0",
    ]


#### CONFIG FOR taylor-green vortex ####
elif case == 'tgv':

    log_file_glob_str = 'log*'
    force_file_glob_strs = []
    history_file_glob_strs = ['*.eny', 'box.his']
    path_to_directories = "/home/henrik/Documents/simulation_data/codeVerification/taylorgreenvortex/"
    save_directory = "/home/henrik/Documents/simulation_data/codeVerification/taylorgreenvortex/figures/"


    directory_names = [
        # "physics/test-history-box/",
        "physics/dns/nelmts-x-64/p8-highre/",
        "physics/dns/nelmts-x-64/p8-weak-highre/",
        # "physics/dns/p4/",
        # "physics/dns/p6/",
        # "physics/dns/p8/",
        # "physics/dns/p10/",
        # "physics/dns/p12/",
        # "physics/dns/p14/",
        # "physics/dns/p16/",
        # "physics/dns/p6/",
        # "physics/dns/p8/",
        # "physics/dns/p10/",
        # "physics/dns/p6-weak/",
        # "physics/dns/p8-weak/",
        # "physics/dns/p10-weak/",
        ]

    # Skip initial or final n points for each (individual/subdir!) force file
    # Note that a small overlap removes the pressure kick after restarting
    # We found that this improves PSDs
    force_file_skip_start = 0
    force_file_skip_end = 0


    # case-specific definitions
    ylabels = ["$C_d$", "$C_l$", "$C_s$"]#, "$C_S$", "$C_L$"]
    ynames = ["Drag", "Lift", "Shear"]#, "Sheer", "Lift"]
    customMetrics = ["F1-total", "F2-total", "F3-total"]#, "F2-total", "F3-total"]

    # Characteristic velocity and lengths
    ref_velocity = 1.0 # non-dimensional velocity [m/s]
    ref_length = 1
    ctu_len = ref_length / ref_velocity # [s]
    ref_area = ref_length**2 # reference area

    # reference time step size (CFL ~= 1)
    dtref = 5e-4

    # Boundary data naming:
    # All periodic boundaries
    boundary_names = [
        # "b0",
    ]
else:
    print(f"Unknown case choice: {case}. Exiting.")
    exit()
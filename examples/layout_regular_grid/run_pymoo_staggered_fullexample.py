import numpy as np
import argparse
import matplotlib.pyplot as plt
from iwopy.interfaces.pymoo import Optimizer_pymoo

import foxes
from foxes_opt.problems.layout import RegularLayoutOptProblem

# StaggeredLayoutOptProblem,
from foxes_opt.problems.layout.geom_layouts.constraints import CFixN
from foxes_opt.objectives import MaxFarmPower
import foxes.variables as FV

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-nt", "--n_t", help="The number of turbines", type=int, default=291)
    parser.add_argument(
        "-t",
        "--turbine_file",
        help="The P-ct-curve csv file (path or static)",
        default="IEA-15MW-D240-H150.csv",
    )
    parser.add_argument(
        "-s",
        "--states",
        help="The states input file (path or static)",
        default="wind_rose_bremen.csv",
    )
    parser.add_argument("-r", "--rotor", help="The rotor model", default="centre")
    parser.add_argument(
        "-w",
        "--wakes",
        help="The wake models",
        default=["Bastankhah2014_linear_k004"],
        nargs="+",
    )
    parser.add_argument("-p", "--pwakes", help="The partial wakes model", default=None)
    # parser.add_argument("--ws", help="The wind speed", type=float, default=9.0)
    # parser.add_argument("--wd", help="The wind direction", type=float, default=270.0)
    parser.add_argument("--ti", help="The TI value", type=float, default=0.08)
    parser.add_argument("--rho", help="The air density", type=float, default=1.225)
    parser.add_argument(
        "-d",
        "--min_dist",
        help="Minimal turbine distance in m",
        type=float,
        default=3 * 240,
    )
    parser.add_argument("-A", "--opt_algo", help="The pymoo algorithm name", default="GA")
    parser.add_argument("-P", "--n_pop", help="The population size", type=int, default=100)
    parser.add_argument("-G", "--n_gen", help="The number of generations", type=int, default=100)
    parser.add_argument("-nop", "--no_pop", help="Switch off vectorization", action="store_true")
    parser.add_argument("-e", "--engine", help="The engine", default="process")

    parser.add_argument("-n", "--n_cpus", help="The number of cpus", default=128, type=int)

    parser.add_argument(
        "-c",
        "--chunksize_states",
        help="The chunk size for states",
        default=None,
        type=int,
    )
    parser.add_argument(
        "-C",
        "--chunksize_points",
        help="The chunk size for points",
        default=None,
        type=int,
    )
    args = parser.parse_args()

    mbook = foxes.models.ModelBook()
    ttype = foxes.models.turbine_types.PCtFile(args.turbine_file)
    mbook.turbine_types[ttype.name] = ttype

    area = np.array(
        [
            [27651.45772787, -19628.02043291],
            [22467.63504098, -16150.80954337],
            [3382.2115456, -3348.05276415],
            [310.50081837, -308.21665244],
            [0.0, 0.0],
            [11975.95707281, 7327.67969099],
            [17303.90920729, 8551.1474107],
            [18241.37854828, 7390.91648652],
            [23809.67306438, 493.02511296],
            [27325.31528285, -3882.41013356],
            [28927.77174086, -5885.58277772],
        ]
    )

    boundary = foxes.utils.geom2d.ClosedPolygon(area)

    farm = foxes.WindFarm(boundary=boundary)
    farm.add_turbine(
        foxes.Turbine(xy=np.array([16812.74743472, -3981.70549441]), turbine_models=["layout_opt", ttype.name])
    )

    states = foxes.input.states.StatesTable(
        data_source=args.states,
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        var2col={FV.WS: "ws", FV.WD: "wd", FV.WEIGHT: "weight"},
        fixed_vars={FV.RHO: args.rho, FV.TI: args.ti},
    )

    algo = foxes.algorithms.Downwind(
        farm,
        states,
        rotor_model=args.rotor,
        wake_models=args.wakes,
        wake_frame="rotor_wd",
        partial_wakes=args.pwakes,
        mbook=mbook,
        verbosity=0,
    )

    print("Using engine:", args.engine)
    with foxes.Engine.new(
        engine_type=args.engine,
        n_procs=args.n_cpus,
        chunk_size_states=args.chunksize_states,
        chunk_size_points=args.chunksize_points,
        verbosity=1,
    ):
        #

        try:  # staggered layout
            problem = RegularLayoutOptProblem("layout_opt", algo, min_spacing=args.min_dist, staggered=True)
        except:
            print(
                "Do not have staggered layout implemented yet, change for newer version. Will continue with standard Regular layout"
            )
            problem = RegularLayoutOptProblem("layout_opt", algo, min_spacing=args.min_dist)

        problem.add_objective(MaxFarmPower(problem))
        problem.add_constraint(CFixN(problem, args.n_t))

        problem.initialize()

        solver = Optimizer_pymoo(
            problem,
            problem_pars=dict(
                vectorize=not args.no_pop,
            ),
            algo_pars=dict(
                type=args.opt_algo,
                pop_size=args.n_pop,
                seed=None,
            ),
            setup_pars=dict(),
            term_pars=("n_gen", args.n_gen),
        )
        solver.initialize()
        solver.print_info()

        # print("will show figure")
        # ax = foxes.output.FarmLayoutOutput(farm).get_figure()
        # plt.show()
        # plt.close(ax.get_figure())
        # print("Shown figure")

        results = solver.solve()
        solver.finalize(results)

        print()
        print(results)

        fig, axs = plt.subplots(1, 2, figsize=(12, 8))

        foxes.output.FarmLayoutOutput(farm).get_figure(fig=fig, ax=axs[0])

        o = foxes.output.FlowPlots2D(algo, results.problem_results)
        plt.show()
        plt.savefig("layout_regular_grid_staggered.png", dpi=300)

        p_min = np.array([-1100.0, -1100.0])
        p_max = np.array([1100.0, 2000.0])

        fig = o.get_mean_fig_xy(
            "WS",
            resolution=20,
            fig=fig,
            ax=axs[1],
            xmin=p_min[0],
            xmax=p_max[0],
            ymin=p_min[1],
            ymax=p_max[1],
        )
        dpars = dict(alpha=0.6, zorder=10, p_min=p_min, p_max=p_max)
        farm.boundary.add_to_figure(axs[1], fill_mode="outside_white", pars_distance=dpars)

        plt.show()
        plt.savefig("MeanWSField.png", dpi=300)
        plt.close(fig)

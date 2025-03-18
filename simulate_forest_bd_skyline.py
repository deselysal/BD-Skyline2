import logging

import numpy as np

from treesimulator import save_forest, save_log, save_ltt
from treesimulator.generator_skyline import generate, observed_ltt
from treesimulator.mtbd_models import BirthDeathModel, CTModel


def main():
    """
    Entry point for tree/forest generation with the BD model with command-line arguments.
    :return: void
    """
    import argparse

    parser = \
        argparse.ArgumentParser(description="Simulates a tree (or a forest of trees) for given BD model parameters. "
                                            "If a simulation leads to less than --min_tips tips, it is repeated.")
    parser.add_argument('--min_tips', required=True, type=int,
                        help="desired minimal bound on the total number of simulated leaves. "
                             "For a tree simulation, "
                             "if --min_tips and --max_tips are equal, exactly that number of tips will be simulated. "
                             "If --min_tips is less than --max_tips, "
                             "a value randomly drawn between one and another will be simulated.")
    parser.add_argument('--max_tips', required=True, type=int,
                        help="desired maximal bound on the total number of simulated leaves"
                             "For a tree simulation, "
                             "if --min_tips and --max_tips are equal, exactly that number of tips will be simulated. "
                             "If --min_tips is less than --max_tips, "
                             "a value randomly drawn between one and another will be simulated.")
    parser.add_argument('--T', required=False, default=np.inf, type=float,
                        help="Total simulation time. If specified, a forest will be simulated instead of one tree. "
                             "The trees in this forest will be simulated during the given time, "
                             "till the --min_tips number is reached. If after simulating the last tree, "
                             "the forest exceeds the --max_tips number, the process will be restarted.")
    parser.add_argument('--la', default=[0.4, 0.5, 0.6], nargs='+', type=float,
                        help="List of transmission rates for each interval.")
    parser.add_argument('--psi', default=[0.1, 0.2, 0.3], nargs='+', type=float,
                        help="List of removal rates for each interval.")
    parser.add_argument('--p', default=[0.5, 0.6, 0.7], nargs='+', type=float,
                        help="List of sampling probabilities for each interval.")
    parser.add_argument('--times', default=[2.0, 5.0, 10.0], nargs='+', type=float,
                        help="List of times for each interval transition.")
    parser.add_argument('--upsilon', required=False, default=0, type=float, help='notification probability')
    parser.add_argument('--max_notified_contacts', required=False, default=1, type=int,
                        help='maximum number of notified contacts per person')
    parser.add_argument('--avg_recipients', required=False, default=1, type=float,
                        help='average number of recipients per transmission. '
                             'By default one (one-to-one transmission), '
                             'but if a larger number is given then one-to-many transmissions become possible.')
    parser.add_argument('--phi', required=False, default=0, type=float, help='notified removal rate')
    parser.add_argument('--log', required=True, type=str, help="output log file")
    parser.add_argument('--nwk', required=True, type=str, help="output tree or forest file")
    parser.add_argument('--ltt', required=False, default=None, type=str, help="output LTT file")
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help="describe generation process")
    params = parser.parse_args()
    logging.getLogger().handlers = []
    logging.basicConfig(level=logging.DEBUG if params.verbose else logging.INFO,
                        format='%(asctime)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    is_mult = params.avg_recipients != 1
    logging.info('BD{} skyline model parameters are:\n\tlambda={}\n\tpsi={}\n\tp={}\n\ttimes={}{}'
                 .format('-MULT' if is_mult else '',
                         params.la, params.psi, params.p, params.times,
                         '\n\tr={}'.format(params.avg_recipients) if is_mult else ''))

    # Create a list of models with their associated transition times
    models = []
    skyline_times = []

    # Ensure all parameter lists have the same length
    num_intervals = min(len(params.la), len(params.psi), len(params.p), len(params.times))

    for i in range(num_intervals):
        la = params.la[i] if i < len(params.la) else params.la[-1]
        psi = params.psi[i] if i < len(params.psi) else params.psi[-1]
        p = params.p[i] if i < len(params.p) else params.p[-1]
        time = params.times[i] if i < len(params.times) else params.times[-1]

        model = BirthDeathModel(p=p, la=la, psi=psi, n_recipients=[params.avg_recipients])

        if params.upsilon and params.upsilon > 0:
            if i == 0:  # Log this only once
                logging.info('PN parameters are:\n\tphi={}\n\tupsilon={}'.format(params.phi, params.upsilon))
            model = CTModel(model=model, upsilon=params.upsilon, phi=params.phi)

        models.append(model)
        skyline_times.append(time)

        logging.info(f'Model {i + 1} with transition time {time}: lambda={la}, psi={psi}, p={p}')

    if params.T < np.inf:
        logging.info('Total time T={}'.format(params.T))

    forest, (total_tips, u, T), ltt = generate(models, params.min_tips, params.max_tips, T=params.T,
                                               skyline_times=skyline_times,
                                               max_notified_contacts=params.max_notified_contacts)

    save_forest(forest, params.nwk)

    # For log saving, use the last model for now (could be improved)
    save_log(models[-1], total_tips, T, u, params.log)

    if params.ltt:
        save_ltt(ltt, observed_ltt(forest, T), params.ltt)


if '__main__' == __name__:
    main()
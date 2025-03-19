import logging
import numpy as np
from treesimulator import save_forest, save_log, save_ltt
from treesimulator.generator import generate, observed_ltt
from treesimulator.mtbd_models import BirthDeathModel, CTModel


def main():
    """
    Entry point for tree/forest generation with the BD-Skyline model using a list-based approach.
    :return: void
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Simulates a tree (or a forest of trees) with the BD-Skyline approach using a list of models.")

    parser.add_argument('--min_tips', default=5, type=int, help="Minimum number of simulated leaves.")
    parser.add_argument('--max_tips', default=20, type=int, help="Maximum number of simulated leaves.")
    parser.add_argument('--T', required=False, default=np.inf, type=float, help="Total simulation time.")
    parser.add_argument('--la', default=[0.4, 0.5, 0.6], nargs='+', type=float,
                        help="List of transmission rates for each interval.")
    parser.add_argument('--psi', default=[0.1, 0.2, 0.3], nargs='+', type=float,
                        help="List of removal rates for each interval.")
    parser.add_argument('--p', default=[0.5, 0.6, 0.7], nargs='+', type=float,
                        help="List of sampling probabilities for each interval.")
    parser.add_argument('--t', default=[2.0, 5.0, 10.0], nargs='+', type=float,
                        help="List of time points corresponding to parameters change.")
    parser.add_argument('--upsilon', required=False, default=0, type=float, help='Notification probability')
    parser.add_argument('--max_notified_contacts', required=False, default=1, type=int,
                        help='Maximum notified contacts')
    parser.add_argument('--avg_recipients', required=False, default=1, type=float,
                        help='Average number of recipients per transmission.')
    parser.add_argument('--log', default='output.log', type=str, help="Output log file")
    parser.add_argument('--nwk', default='output.nwk', type=str, help="Output tree or forest file")
    parser.add_argument('--ltt', required=False, default=None, type=str, help="Output LTT file")
    parser.add_argument('-v', '--verbose', default=True, action='store_true', help="Verbose output")

    params = parser.parse_args()
    logging.getLogger().handlers = []
    logging.basicConfig(level=logging.DEBUG if params.verbose else logging.INFO,
                        format='%(asctime)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    # Validate parameters
    if len(params.la) != len(params.psi) or len(params.la) != len(params.p):
        raise ValueError("Parameters la, psi, and p must have the same length")

    # For skyline model, parameter count should match time points count
    # In the original BDSkylineIMproved, each parameter corresponds to a time point
    if len(params.t) != len(params.la):
        raise ValueError(
            f"For skyline models, the number of parameter sets must equal the number of time points. Got {len(params.la)} parameter sets and {len(params.t)} time points.")

    # Log the configuration
    logging.info('BD-Skyline parameters are:')
    logging.info(f'Lambda values: {params.la}')
    logging.info(f'Psi values: {params.psi}')
    logging.info(f'p values: {params.p}')
    logging.info(f'Time points: {params.t}')

    # Create a list of BirthDeath models
    models = []
    for i in range(len(params.la)):
        model_name = f'BD{i + 1}'
        logging.info(f'Creating model {model_name} with la={params.la[i]}, psi={params.psi[i]}, p={params.p[i]}')

        model = BirthDeathModel(la=params.la[i], psi=params.psi[i], p=params.p[i],
                                n_recipients=[params.avg_recipients])

        if params.upsilon and params.upsilon > 0:
            model = CTModel(model=model, upsilon=params.upsilon)

        models.append(model)

    if params.T < np.inf:
        logging.info(f'Total time T={params.T}')

    # Generate forest using the skyline model approach
    try:
        forest, (total_tips, u, T), ltt = generate(
            models,
            min_tips=params.min_tips,
            max_tips=params.max_tips,
            T=params.T,
            skyline_times=params.t,  # Pass time points for model changes
            max_notified_contacts=params.max_notified_contacts
        )

        # Save outputs
        save_forest(forest, params.nwk)
        # For logging, use the first model (without the skyline parameter)
        save_log(models[0], total_tips, T, u, params.log)
        if params.ltt:
            save_ltt(ltt, observed_ltt(forest, T), params.ltt)

        logging.info("Simulation completed successfully")

    except RuntimeError as e:
        logging.error(f"Runtime error during simulation: {e}")
    except ValueError as e:
        logging.error(f"Value error during simulation: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")


if '__main__' == __name__:
    main()
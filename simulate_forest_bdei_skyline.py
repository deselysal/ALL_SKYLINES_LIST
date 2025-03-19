import logging
import numpy as np
from treesimulator import save_forest, save_log, save_ltt
from treesimulator.generator import generate, observed_ltt
from treesimulator.mtbd_models import BirthDeathExposedInfectiousModel, CTModel


def main():
    """
    Entry point for tree/forest generation using the BDEI-Skyline model with command-line arguments.
    Now implemented using a list-based approach rather than a dedicated skyline class.
    :return: void
    """
    # Set up logging configuration to match the BDEI example
    logging.getLogger().handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    import argparse

    parser = argparse.ArgumentParser(
        description="Simulates a tree (or a forest of trees) with the BDEI-Skyline model using a list-based approach.")

    # Default values are added for each parameter
    parser.add_argument('--min_tips', default=5, type=int, help="Minimum number of simulated leaves.")
    parser.add_argument('--max_tips', default=20, type=int, help="Maximum number of simulated leaves.")
    parser.add_argument('--T', required=False, default=np.inf, type=float, help="Total simulation time.")
    parser.add_argument('--mu', default=[0.05, 0.06], nargs='+', type=float,
                        help="List of E->I transition rates for each interval.")
    parser.add_argument('--la', default=[0.4, 0.5], nargs='+', type=float,
                        help="List of transmission rates for each interval.")
    parser.add_argument('--psi', default=[0.1, 0.2], nargs='+', type=float,
                        help="List of removal rates for each interval.")
    parser.add_argument('--p', default=[0.5, 0.6], nargs='+', type=float,
                        help="List of sampling probabilities for each interval.")
    parser.add_argument('--t', default=[2.0, 5.0], nargs='+', type=float,
                        help="List of time points corresponding to parameters change.")
    parser.add_argument('--upsilon', required=False, default=0, type=float, help='Notification probability')
    parser.add_argument('--max_notified_contacts', required=False, default=1, type=int,
                        help='Maximum notified contacts')
    parser.add_argument('--avg_recipients', required=False, default=1, type=float,
                        help='Average number of recipients per transmission.')
    parser.add_argument('--log', default='output.log', type=str, help="Output log file")
    parser.add_argument('--nwk', default='output.nwk', type=str, help="Output tree or forest file")
    parser.add_argument('--ltt', required=False, default=None, type=str, help="Output LTT file")
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help="Verbose output")

    params = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if params.verbose else logging.INFO,
                        format='%(asctime)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    try:
        # Validate parameters
        if len(params.mu) != len(params.la) or len(params.mu) != len(params.psi) or len(params.mu) != len(params.p):
            raise ValueError("Parameters mu, la, psi, and p must have the same length")

        # For skyline model, parameter count should match time points count
        if len(params.t) != len(params.mu):
            raise ValueError(
                f"For skyline models, the number of parameter sets must equal the number of time points. Got {len(params.mu)} parameter sets and {len(params.t)} time points.")

        # Log the configuration
        is_mult = params.avg_recipients != 1
        mult_str = '-MULT' if is_mult else ''
        logging.info(f'BDEI{mult_str}-Skyline parameters are:')
        logging.info(f'mu values: {params.mu}')
        logging.info(f'lambda values: {params.la}')
        logging.info(f'psi values: {params.psi}')
        logging.info(f'p values: {params.p}')
        logging.info(f'Time points: {params.t}')
        if is_mult:
            logging.info(f'Average recipients: {params.avg_recipients}')

        # Create a list of BDEI models
        models = []
        for i in range(len(params.mu)):
            model_name = f'BDEI{i + 1}'
            logging.info(
                f'Creating model {model_name} with mu={params.mu[i]}, la={params.la[i]}, psi={params.psi[i]}, p={params.p[i]}')

            # Create a BDEI model with the parameters for this time interval
            model = BirthDeathExposedInfectiousModel(
                mu=params.mu[i],
                la=params.la[i],
                psi=params.psi[i],
                p=params.p[i],
                n_recipients=[params.avg_recipients, params.avg_recipients]
            )

            # Apply contact tracing if specified
            if params.upsilon and params.upsilon > 0:
                model = CTModel(model=model, upsilon=params.upsilon)
                logging.info(f'Added contact tracing with upsilon={params.upsilon}')

            models.append(model)

        if params.T < np.inf:
            logging.info(f'Total time T={params.T}')

        # Generate forest using the skyline model approach (list of models)
        forest, (total_tips, u, max_time), ltt = generate(
            models,
            min_tips=params.min_tips,
            max_tips=params.max_tips,
            T=params.T,
            skyline_times=params.t,  # Pass time points for model changes
            max_notified_contacts=params.max_notified_contacts
        )

        # Save outputs
        save_forest(forest, params.nwk)
        # For logging, use the first model (without the skyline parameter which isn't supported)
        save_log(models[0], total_tips, max_time, u, params.log)
        if params.ltt:
            save_ltt(ltt, observed_ltt(forest, max_time), params.ltt)

        logging.info("Simulation completed successfully")

    except RuntimeError as e:
        logging.error(f"Runtime error during simulation: {e}")
    except ValueError as e:
        logging.error(f"Value error during simulation: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")


if '__main__' == __name__:
    main()
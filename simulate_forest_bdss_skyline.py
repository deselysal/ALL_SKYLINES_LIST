import logging
import numpy as np
from treesimulator import save_forest, save_log, save_ltt
from treesimulator.generator import generate, observed_ltt
from treesimulator.mtbd_models import BirthDeathWithSuperSpreadingModel, CTModel


def main():
    """
    Entry point for tree/forest generation using the BDSS-Skyline model with command-line arguments.
    Now implemented using a list-based approach rather than a dedicated skyline class.
    :return: void
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Simulates a tree (or a forest of trees) with the BDSS-Skyline model using a list-based approach.")

    parser.add_argument('--min_tips', default=5, type=int, help="Minimum number of simulated leaves.")
    parser.add_argument('--max_tips', default=20, type=int, help="Maximum number of simulated leaves.")
    parser.add_argument('--T', required=False, default=np.inf, type=float, help="Total simulation time.")
    parser.add_argument('--la_nn', default=[0.3, 0.4, 0.5], nargs='+', type=float,
                        help="List of transmission rates from normal to normal for each interval.")
    parser.add_argument('--la_ns', default=[0.1, 0.15, 0.2], nargs='+', type=float,
                        help="List of transmission rates from normal to super for each interval.")
    parser.add_argument('--la_sn', default=[0.6, 0.8, 1.0], nargs='+', type=float,
                        help="List of transmission rates from super to normal for each interval.")
    parser.add_argument('--la_ss', default=[0.2, 0.3, 0.4], nargs='+', type=float,
                        help="List of transmission rates from super to super for each interval.")
    parser.add_argument('--psi', default=[0.1, 0.2, 0.3], nargs='+', type=float,
                        help="List of removal rates for each interval.")
    parser.add_argument('--p', default=[0.5, 0.6, 0.7], nargs='+', type=float,
                        help="List of sampling probabilities for normal spreaders for each interval.")
    parser.add_argument('--p_s', default=[0.5, 0.6, 0.7], nargs='+', type=float,
                        help="List of sampling probabilities for superspreaders for each interval.")
    parser.add_argument('--t', default=[2.0, 5.0, 10.0], nargs='+', type=float,
                        help="List of time points corresponding to parameters change.")
    parser.add_argument('--upsilon', required=False, default=0, type=float, help='Notification probability')
    parser.add_argument('--max_notified_contacts', required=False, default=1, type=int,
                        help='Maximum notified contacts')
    parser.add_argument('--avg_recipients', nargs=2, default=[1, 1], type=float,
                        help='Average number of recipients per transmission for each donor state (normal spreader, superspreader).')
    parser.add_argument('--log', default='output.log', type=str, help="Output log file")
    parser.add_argument('--nwk', default='output.nwk', type=str, help="Output tree or forest file")
    parser.add_argument('--ltt', required=False, default=None, type=str, help="Output LTT file")
    parser.add_argument('-v', '--verbose', default=False, action='store_true', help="Verbose output")

    params = parser.parse_args()
    logging.getLogger().handlers = []
    logging.basicConfig(level=logging.DEBUG if params.verbose else logging.INFO,
                        format='%(asctime)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    try:
        # Check if all parameter arrays have the same length
        param_lengths = [
            len(params.la_nn), len(params.la_ns), len(params.la_sn), len(params.la_ss),
            len(params.psi), len(params.p), len(params.p_s), len(params.t)
        ]

        if len(set(param_lengths)) > 1:
            raise ValueError("All parameter arrays must have the same length!")

        # For skyline model, parameter count should match time points count
        if len(params.t) != len(params.la_nn):
            raise ValueError(
                f"For skyline models, the number of parameter sets must equal the number of time points. Got {len(params.la_nn)} parameter sets and {len(params.t)} time points.")

        # Log the configuration
        is_mult = np.any(np.array(params.avg_recipients) != 1)
        mult_str = '-MULT' if is_mult else ''
        logging.info(f'BDSS{mult_str}-Skyline parameters are:')
        logging.info(f'la_nn values: {params.la_nn}')
        logging.info(f'la_ns values: {params.la_ns}')
        logging.info(f'la_sn values: {params.la_sn}')
        logging.info(f'la_ss values: {params.la_ss}')
        logging.info(f'psi values: {params.psi}')
        logging.info(f'p values: {params.p}')
        logging.info(f'p_s values: {params.p_s}')
        logging.info(f'Time points: {params.t}')
        if is_mult:
            logging.info(f'Average recipients: normal={params.avg_recipients[0]}, super={params.avg_recipients[1]}')

        # Create a list of BDSS models
        models = []
        for i in range(len(params.la_nn)):
            model_name = f'BDSS{i + 1}'

            # Check transmission ratio constraint for this interval
            if params.la_ns[i] == 0 or params.la_nn[i] == 0:
                # If either denominator is zero, check if all are consistent with zero
                if (params.la_ns[i] == 0 and params.la_ss[i] == 0) or (params.la_nn[i] == 0 and params.la_sn[i] == 0):
                    pass  # This is fine, both ratios are effectively 0/0 which we'll treat as equal
                else:
                    raise ValueError(
                        f'Transmission ratio constraint cannot be verified for interval {i + 1}: Cannot divide by zero '
                        f'(la_ns={params.la_ns[i]}, la_nn={params.la_nn[i]})'
                    )
            else:
                s_ratio = params.la_ss[i] / params.la_ns[i]
                n_ratio = params.la_sn[i] / params.la_nn[i]
                if np.abs(s_ratio - n_ratio) > 1e-3:
                    raise ValueError(
                        f'Transmission ratio constraint is violated for interval {i + 1}: '
                        f'la_ss / la_ns ({s_ratio}) must be equal to la_sn / la_nn ({n_ratio})'
                    )

            logging.info(f'Creating model {model_name} with:')
            logging.info(
                f'  la_nn={params.la_nn[i]}, la_ns={params.la_ns[i]}, la_sn={params.la_sn[i]}, la_ss={params.la_ss[i]}')
            logging.info(f'  psi={params.psi[i]}, p={params.p[i]}, p_s={params.p_s[i]}')

            # Create a BDSS model with the parameters for this time interval
            model = BirthDeathWithSuperSpreadingModel(
                la_nn=params.la_nn[i],
                la_ns=params.la_ns[i],
                la_sn=params.la_sn[i],
                la_ss=params.la_ss[i],
                psi=params.psi[i],
                p=params.p[i],
                p_s=params.p_s[i],
                n_recipients=params.avg_recipients
            )

            # Apply contact tracing if specified
            if params.upsilon and params.upsilon > 0:
                model = CTModel(model=model, upsilon=params.upsilon)
                logging.info(f'Added contact tracing with upsilon={params.upsilon}')

            models.append(model)

        if params.T < np.inf:
            logging.info(f'Total time T={params.T}')

        # Generate forest using the skyline model approach (list of models)
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
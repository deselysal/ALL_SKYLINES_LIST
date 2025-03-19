import logging

import numpy as np

from treesimulator import save_forest, save_log, save_ltt
from treesimulator.generator import generate, observed_ltt
from treesimulator.mtbd_models import Model, CTModel


def main():
    """
    Entry point for tree/forest generation with a generic MTBD model, supporting both
    standard and skyline (time-varying) models.
    :return: void
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Simulates a tree (or a forest of trees) for given MTBD model parameters. "
                    "Supports both standard models and skyline (time-varying) models. "
                    "If a simulation leads to less than --min_tips tips, it is repeated.")

    # Common parameters
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

    # Model parameters - now with support for multiple time points
    parser.add_argument('--states', nargs='+', type=str, help="model states")
    parser.add_argument('--transition_rates', nargs='+', type=float, action='append',
                        help="transition rate matrix for each time interval, row after row, in the same order as model states, "
                             "e.g. if a model has 2 states given as --states E I,"
                             "then here we expect E->E E->I I->E I->I for each time interval. "
                             "To provide multiple matrices for a skyline model, use this flag multiple times.")
    parser.add_argument('--transmission_rates', nargs='+', type=float, action='append',
                        help="transmission rate matrix for each time interval, row after row, in the same order as model states, "
                             "e.g. if a model has 2 states given as --states E I,"
                             "then here we expect E->E E->I I->E I->I for each time interval. "
                             "To provide multiple matrices for a skyline model, use this flag multiple times.")
    parser.add_argument('--removal_rates', nargs='+', type=float, action='append',
                        help="removal rate array for each time interval, in the same order as model states, "
                             "e.g. if a model has 2 states given as --states E I,"
                             "then here we expect removal(E) removal(I) for each time interval. "
                             "To provide multiple arrays for a skyline model, use this flag multiple times.")
    parser.add_argument('--sampling_probabilities', nargs='+', type=float, action='append',
                        help="sampling probability array for each time interval, in the same order as model states, "
                             "e.g. if a model has 2 states given as --states E I,"
                             "then here we expect p(E) p(I) for each time interval. "
                             "To provide multiple arrays for a skyline model, use this flag multiple times.")
    parser.add_argument('--t', nargs='+', type=float,
                        help="time points for skyline models, indicating when parameter sets change. "
                             "Required when using multiple parameter sets.")

    # Contact tracing parameters
    parser.add_argument('--upsilon', required=False, default=0, type=float, help='notification probability')
    parser.add_argument('--phi', required=False, default=0, type=float, help='notified removal rate')
    parser.add_argument('--max_notified_contacts', required=False, default=1, type=int,
                        help='maximum number of notified contracts per person')

    # Averaging parameters
    parser.add_argument('--avg_recipients', nargs='*', type=float,
                        help='average number of recipients per transmission '
                             'for each donor state (in the same order as the model states). '
                             'By default, only one-to-one transmissions are allowed, '
                             'but if larger numbers are given then one-to-many transmissions become possible.')

    # Output parameters
    parser.add_argument('--log', required=True, type=str, help="output log file")
    parser.add_argument('--nwk', required=True, type=str, help="output tree or forest file")
    parser.add_argument('--ltt', required=False, default=None, type=str, help="output LTT file")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="print information on the progress of the tree generation (to console)")

    params = parser.parse_args()
    logging.getLogger().handlers = []
    logging.basicConfig(level=logging.DEBUG if params.verbose else logging.INFO,
                        format='%(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    n_states = len(params.states)

    # Check if we're using a skyline model (multiple parameter sets)
    use_skyline = (isinstance(params.transition_rates, list) and len(params.transition_rates) > 1 or
                   isinstance(params.transmission_rates, list) and len(params.transmission_rates) > 1 or
                   isinstance(params.removal_rates, list) and len(params.removal_rates) > 1 or
                   isinstance(params.sampling_probabilities, list) and len(params.sampling_probabilities) > 1)

    # Validate skyline parameters
    if use_skyline:
        # Ensure time points are provided
        if params.t is None:
            raise ValueError(
                "Time points (--t) must be provided when using multiple parameter sets for a skyline model")

        # Check that all parameter arrays have the same number of time points
        param_lengths = []
        if params.transition_rates: param_lengths.append(len(params.transition_rates))
        if params.transmission_rates: param_lengths.append(len(params.transmission_rates))
        if params.removal_rates: param_lengths.append(len(params.removal_rates))
        if params.sampling_probabilities: param_lengths.append(len(params.sampling_probabilities))

        if len(set(param_lengths)) > 1:
            raise ValueError("All parameter arrays must have the same number of time points for a skyline model")

        if params.t and len(params.t) != param_lengths[0]:
            raise ValueError(
                f"Number of time points ({len(params.t)}) must match the number of parameter sets ({param_lengths[0]})")

        logging.info(f"Creating a skyline model with {param_lengths[0]} time intervals at time points: {params.t}")

    # Handle default values for avg_recipients
    if not params.avg_recipients:
        params.avg_recipients = [1] * n_states

    is_mult = np.any(np.array(params.avg_recipients) != 1)

    # Create either a single model or a list of models for skyline simulation
    models = []

    if use_skyline:
        # Create a separate model for each time interval
        for i in range(len(params.t)):
            # Get parameters for this time interval
            transition_matrix = np.array(
                params.transition_rates[i] if i < len(params.transition_rates) else params.transition_rates[
                    -1]).reshape((n_states, n_states))
            transmission_matrix = np.array(
                params.transmission_rates[i] if i < len(params.transmission_rates) else params.transmission_rates[
                    -1]).reshape((n_states, n_states))
            removal_rates = params.removal_rates[i] if i < len(params.removal_rates) else params.removal_rates[-1]
            sampling_probs = params.sampling_probabilities[i] if i < len(params.sampling_probabilities) else \
            params.sampling_probabilities[-1]

            logging.info(f"Model {i + 1} at time < {params.t[i]}:")
            logging.info(f'\ttransition_rates=\n{transition_matrix}')
            logging.info(f'\ttransmission_rates=\n{transmission_matrix}')
            logging.info(f'\tremoval_rates={removal_rates}')
            logging.info(f'\tsampling probabilities={sampling_probs}')
            if is_mult:
                logging.info(f'\tavg_recipient_numbers={params.avg_recipients}')

            # Create model for this interval
            model = Model(states=params.states,
                          transmission_rates=transmission_matrix,
                          transition_rates=transition_matrix,
                          removal_rates=removal_rates,
                          ps=sampling_probs,
                          n_recipients=params.avg_recipients)

            # Apply contact tracing if specified
            if params.upsilon and params.upsilon > 0:
                model = CTModel(model=model, upsilon=params.upsilon, phi=params.phi)

            models.append(model)

    else:
        # Single model case - backward compatibility
        transition_matrix = np.array(params.transition_rates[0] if isinstance(params.transition_rates[0],
                                                                              list) else params.transition_rates).reshape(
            (n_states, n_states))
        transmission_matrix = np.array(params.transmission_rates[0] if isinstance(params.transmission_rates[0],
                                                                                  list) else params.transmission_rates).reshape(
            (n_states, n_states))
        removal_rates = params.removal_rates[0] if isinstance(params.removal_rates[0], list) else params.removal_rates
        sampling_probs = params.sampling_probabilities[0] if isinstance(params.sampling_probabilities[0],
                                                                        list) else params.sampling_probabilities

        logging.info(
            'MTBD{} model parameters are:\n\ttransition_rates=\n{}\n\ttransmission_rates=\n{}\n\tremoval_rates={}\n\tsampling probabilities={}{}'
            .format('-MULT' if is_mult else '',
                    transition_matrix, transmission_matrix, removal_rates,
                    sampling_probs,
                    '\n\tavg_recipient_numbers={}'.format(params.avg_recipients) if is_mult else ''))

        model = Model(states=params.states,
                      transmission_rates=transmission_matrix,
                      transition_rates=transition_matrix,
                      removal_rates=removal_rates,
                      ps=sampling_probs,
                      n_recipients=params.avg_recipients)

        if params.upsilon and params.upsilon > 0:
            logging.info('PN parameters are:\n\tphi={}\n\tupsilon={}'.format(params.phi, params.upsilon))
            model = CTModel(model=model, upsilon=params.upsilon, phi=params.phi)

        # For the single model case, we still use a list with one element for consistency
        models = [model]

    if params.T < np.inf:
        logging.info('Total time T={}'.format(params.T))

    # Use the generic generator to handle both single and skyline models
    forest, (total_tips, u, T), ltt = generate(
        models,
        params.min_tips,
        params.max_tips,
        T=params.T,
        skyline_times=params.t if use_skyline else None,
        max_notified_contacts=params.max_notified_contacts
    )

    save_forest(forest, params.nwk)
    save_log(models[0], total_tips, T, u, params.log)
    if params.ltt:
        save_ltt(ltt, observed_ltt(forest, T), params.ltt)


if '__main__' == __name__:
    main()
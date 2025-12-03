from time import time

import numpy as np
import matplotlib.pyplot as plt
import wntr
import networkx as nx
from scipy.signal import savgol_filter
# from statsmodels.tsa.seasonal import seasonal_decompose


def split_wn_by_profile(wn: wntr.network.WaterNetworkModel,
                        p_commercial: float):
    graph = wn.to_graph(link_weight={k: v for k, v in wn.query_link_attribute('length').items()})
    graph = graph.subgraph(wn.node_name_list).to_undirected()

    communities_generator = nx.community.louvain_communities(graph)
    sorted_communities = sorted(list(communities_generator), key=lambda s: len(s))

    commercial = set()
    for community in sorted_communities:
        if len(commercial) < int(len(wn.junction_name_list) * p_commercial):
            for node in community:
                commercial.add(node)
        else:
            break

    g_comm = graph.subgraph(commercial)
    isolated_comm = list(nx.isolates(g_comm))
    assert len(isolated_comm) == 0, "isolated nodes in the commercial group"

    households = set(wn.node_name_list).difference(commercial)
    g_hh = graph.subgraph(households)
    isolated_hh = list(nx.isolates(g_hh))
    assert len(isolated_hh) == 0, "isolated nodes in the household group"

    households = set(wn.junction_name_list).intersection(households)
    commercial = set(wn.junction_name_list).intersection(commercial)

    return list(households), list(commercial)


def generate_24hour_pattern(profile: tuple[int, int, int, int],
                            num_patterns: int,
                            samples_per_hour: int,
                            low_consumption_upper_bound: float,
                            high_consumption_lower_bound: float):
    """
    Given a consumption profile, it generates a demand pattern for 1 day. The day is divided into 4 time slots
    of 6 hours each. From 00.00 to 06.00, 06.00 to 12.00, 12.00 to 18.00, and 18.00 to 00.00. The consumption
    profile is then created by assigning the consumption ranges to each time slot.
    :param high_consumption_lower_bound: value over which the water consumption is considered high
    :param low_consumption_upper_bound: value under which the water consumption is considered low
    :param profile: 4-tuple that indicates the consumption ranges assigned to each time slot of
             the day. The consumption ranges can be 0:low, 1:medium, 2:high.
    :type profile: tuple[int, int, int, int]
    :param samples_per_hour: number of samples per hour
    :param num_patterns: number of patterns to be generated
    :return: 'num_patterns' time series of 24 hour data
    :rtype: numpy.array
    """

    # # low, medium and high thresholds are chosen based on the 25, 50, and 75 percentile of 168 generated numbers.
    # # Those 168 numbers correspond to 1 week of data sampled at 1-hour interval
    # aux_rand = np.random.random(168)
    # low = np.percentile(aux_rand, q=25).item()
    # high = np.percentile(aux_rand, q=75).item()

    consumption_range = {
        0: (0, low_consumption_upper_bound),
        1: (low_consumption_upper_bound, high_consumption_lower_bound),
        2: (high_consumption_lower_bound, 1)
    }

    # samples_per_hour_tmp = 1 if samples_per_hour < 1 else int(samples_per_hour)

    profile = [consumption_range[p] for p in profile]
    patterns = np.hstack(
        (np.random.uniform(profile[0][0], profile[0][1], size=(num_patterns, samples_per_hour * 6)),
         np.random.uniform(profile[1][0], profile[1][1], size=(num_patterns, samples_per_hour * 6)),
         np.random.uniform(profile[2][0], profile[2][1], size=(num_patterns, samples_per_hour * 6)),
         np.random.uniform(profile[3][0], profile[3][1], size=(num_patterns, samples_per_hour * 6))))

    # if samples_per_hour < 1:
    #     samples_per_day = int(samples_per_hour * 24)
    #     idx = np.random.choice(range(24), samples_per_day)
    #     patterns = patterns[:, idx]

    return patterns


def generate_daily_component(num_patterns: int,
                             samples_per_hour: int,
                             num_samples: int,
                             duration: int,
                             profile: tuple[int, int, int, int],
                             min_noise: float,
                             max_noise: float,
                             low_consumption_upper_bound: float,
                             high_consumption_lower_bound: float):
    # samples_per_hour = int(1 / time_step)
    # num_samples = int(duration * samples_per_hour)

    num_repetitions = np.ceil(duration / 24).astype(int)

    daily_data = generate_24hour_pattern(profile=profile,
                                         num_patterns=num_patterns,
                                         samples_per_hour=samples_per_hour,
                                         low_consumption_upper_bound=low_consumption_upper_bound,
                                         high_consumption_lower_bound=high_consumption_lower_bound)

    locs = np.zeros((num_patterns, 1))
    scales = np.random.uniform(min_noise, max_noise, size=(num_patterns, 1))
    noise = np.random.normal(loc=locs, scale=scales, size=(num_patterns, num_samples))

    daily_data = np.tile(daily_data, num_repetitions)[:, :num_samples] + noise
    daily_data = (np.cos(daily_data) + np.sin(daily_data))

    window_size = samples_per_hour * 3 if samples_per_hour > 1 else 2
    daily_data = savgol_filter(daily_data, int(window_size), 1, axis=1)
    min_val = np.min(daily_data, axis=1).reshape(-1, 1)
    max_val = np.max(daily_data, axis=1).reshape(-1, 1)

    daily_data = (daily_data - min_val) / (max_val - min_val)

    return daily_data


def generate_yearly_component(num_patterns: int,
                              num_samples: int,
                              yearly_pattern_num_harmonics: int,
                              summer_amplitude_range: tuple[int, int],
                              summer_peak: float,
                              max_noise: float):
    hours = np.arange(0, num_samples)

    A = np.random.rand(yearly_pattern_num_harmonics)
    B = A * np.random.rand()

    y = 1
    for n in range(1, len(A)):
        y += A[n] * np.cos(2 * np.pi * n * hours / num_samples) + B[n] * np.sin(2 * np.pi * n * hours / num_samples)

    y = np.hamming(num_samples) * y  # used for smoothing the edges of the Fourier time-series

    C = np.random.uniform(low=summer_amplitude_range[0], high=summer_amplitude_range[1], size=(num_patterns, 1))

    # Add the summer peak effect
    seasonal_effect = C * np.cos(2 * np.pi * (hours - summer_peak) / num_samples)
    yearly_data = y + seasonal_effect + np.random.normal(scale=max_noise, size=num_samples)

    min_val = np.min(yearly_data, axis=1).reshape(-1, 1)
    max_val = np.max(yearly_data, axis=1).reshape(-1, 1)

    yearly_data = (yearly_data - min_val) / (max_val - min_val)
    return yearly_data


def generate_random_pattern(num_patterns: int,
                            samples_per_hour: int,
                            duration: int,
                            profile: tuple[int, int, int, int],
                            min_noise: float,
                            max_noise: float,
                            yearly_component: np.ndarray,
                            low_consumption_upper_bound: float,
                            high_consumption_lower_bound: float):
    # samples_per_hour = 1 / time_step
    num_samples = int(duration * samples_per_hour)

    daily_component = generate_daily_component(num_patterns=num_patterns,
                                               samples_per_hour=samples_per_hour,
                                               num_samples=num_samples,
                                               duration=duration,
                                               profile=profile,
                                               min_noise=min_noise,
                                               max_noise=max_noise,
                                               low_consumption_upper_bound=low_consumption_upper_bound,
                                               high_consumption_lower_bound=high_consumption_lower_bound)

    demand_patterns = daily_component + yearly_component
    min_val = np.min(demand_patterns, axis=1).reshape(-1, 1)
    max_val = np.max(demand_patterns, axis=1).reshape(-1, 1)

    demand_patterns_normalized = (demand_patterns - min_val) / (max_val - min_val)

    return demand_patterns_normalized


def generate_demand(wn: wntr.network.WaterNetworkModel,
                    time_step: float,
                    duration: int,
                    yearly_pattern_num_harmonics: int,
                    summer_amplitude_range: tuple[int, int],
                    summer_start: float,
                    summer_rolling_rate: float,
                    min_p_commercial: float,
                    max_p_commercial: float,
                    profile_household: tuple[int, int, int, int],
                    profile_commercial: tuple[int, int, int, int],
                    profile_extreme: tuple[int, int, int, int],
                    min_noise: float,
                    max_noise: float,
                    zero_dem_rate: float = 0,
                    extreme_dem_rate: float = 0,
                    max_extreme_dem_junctions: int = 2):
    assert duration > time_step, "duration has to be greater than time step"

    samples_per_hour = int(1 / time_step) if time_step <= 1 else 1
    num_samples = int(duration * samples_per_hour)

    p_commercial = np.random.uniform(low=min_p_commercial, high=max_p_commercial, size=1).item()
    hh_junction_names, comm_junction_names = split_wn_by_profile(wn=wn, p_commercial=p_commercial)
    mask_proba = np.random.rand(len(wn.junction_name_list))

    extreme_junctions_names = []
    if extreme_dem_rate > 0:
        mask = mask_proba <= extreme_dem_rate
        extreme_junctions_names = set(np.array(wn.junction_name_list)[mask][:max_extreme_dem_junctions])

    zero_junctions_names = []
    if zero_dem_rate > 0:
        mask = mask_proba <= zero_dem_rate  # 0.12782
        zero_junctions_names = np.array(wn.junction_name_list)[mask]
        zero_junctions_names = set(zero_junctions_names).difference(extreme_junctions_names)

    hh_junction_names = set(hh_junction_names).difference(zero_junctions_names).difference(extreme_junctions_names)
    comm_junction_names = set(comm_junction_names).difference(zero_junctions_names).difference(extreme_junctions_names)

    num_comm_patterns = len(comm_junction_names)
    num_hh_patterns = len(hh_junction_names)
    num_zero_demand = len(zero_junctions_names)
    num_extreme_demand = len(extreme_junctions_names)

    num_patterns = num_hh_patterns + num_comm_patterns + num_zero_demand + num_extreme_demand

    assert num_comm_patterns + num_hh_patterns + num_zero_demand + num_extreme_demand == len(wn.junction_name_list), (
        "Incorrect number of junctions")

    p_roll = np.random.rand()
    if p_roll < summer_rolling_rate:
        summer_start = np.random.choice(np.arange(0, 12) / 12)  # randomly pick the start of a month

    summer_end = summer_start + (3 / 12)  # start of summer + 3 months, assuming 3-month summer duration
    summer_peak = np.random.uniform(summer_start, summer_end)

    if summer_peak > 1:
        summer_peak += -1

    summer_peak = int(summer_peak * samples_per_hour * duration)

    # print(p_roll)
    # print(summer_start, summer_end, summer_peak)

    # low and high thresholds are chosen based on the 25 and 75 percentile of 168 random numbers.
    # Those 168 numbers mimic to 1 week of data sampled at 1-hour interval
    aux_rand = np.random.random(num_samples)
    low_consumption_upper_bound = np.percentile(aux_rand, q=25).item()
    high_consumption_lower_bound = np.percentile(aux_rand, q=75).item()

    yearly_component = generate_yearly_component(num_patterns=num_patterns,
                                                 num_samples=num_samples,
                                                 yearly_pattern_num_harmonics=yearly_pattern_num_harmonics,
                                                 summer_amplitude_range=summer_amplitude_range,
                                                 summer_peak=summer_peak,
                                                 max_noise=max_noise)

    hh_demand = generate_random_pattern(num_patterns=num_hh_patterns,
                                        samples_per_hour=samples_per_hour,
                                        duration=duration,
                                        profile=profile_household,
                                        min_noise=min_noise,
                                        max_noise=max_noise,
                                        yearly_component=yearly_component[:num_hh_patterns],
                                        low_consumption_upper_bound=low_consumption_upper_bound,
                                        high_consumption_lower_bound=high_consumption_lower_bound)

    comm_demand = generate_random_pattern(num_patterns=num_comm_patterns,
                                          samples_per_hour=samples_per_hour,
                                          duration=duration,
                                          profile=profile_commercial,
                                          min_noise=min_noise,
                                          max_noise=max_noise,
                                          yearly_component=yearly_component[
                                                           num_hh_patterns:num_hh_patterns + num_comm_patterns],
                                          low_consumption_upper_bound=low_consumption_upper_bound,
                                          high_consumption_lower_bound=high_consumption_lower_bound)
    # re-scaling of commercial demand
    comm_demand = comm_demand * (1 - low_consumption_upper_bound) + low_consumption_upper_bound

    if num_extreme_demand > 0:
        extreme_demand = generate_random_pattern(num_patterns=num_extreme_demand,
                                                 samples_per_hour=samples_per_hour,
                                                 duration=duration,
                                                 profile=profile_extreme,
                                                 min_noise=min_noise,
                                                 max_noise=max_noise,
                                                 yearly_component=yearly_component[-num_extreme_demand:],
                                                 low_consumption_upper_bound=low_consumption_upper_bound,
                                                 high_consumption_lower_bound=high_consumption_lower_bound)

        # re-scaling of extreme demand
        extreme_demand = extreme_demand * (1 - high_consumption_lower_bound) + high_consumption_lower_bound
    else:
        extreme_demand = np.empty((0, hh_demand.shape[1]))

    # assign zeros to zero_demand pattern
    zero_demand = np.full(shape=(num_zero_demand, hh_demand.shape[1]), fill_value=0)

    combined_keys = np.hstack((list(hh_junction_names),
                               list(comm_junction_names),
                               list(zero_junctions_names),
                               list(extreme_junctions_names)))

    combined_dict = dict(zip(combined_keys, range(num_patterns)))
    nodes_order = np.array([combined_dict[k] for k in wn.junction_name_list])

    gen_demand = np.vstack((hh_demand, comm_demand, zero_demand, extreme_demand))[nodes_order]

    hh_junction_indexes = []
    comm_junction_indexes = []
    zero_junction_indexes = []
    extreme_junction_indexes = []

    for i, n in enumerate(wn.junction_name_list):
        if n in hh_junction_names:
            hh_junction_indexes.append(i)
        elif n in comm_junction_names:
            comm_junction_indexes.append(i)
        elif n in zero_junctions_names:
            zero_junction_indexes.append(i)
        elif n in extreme_junctions_names:
            extreme_junction_indexes.append(i)

    if time_step > 1:
        gen_demand = gen_demand[:, :num_samples:time_step][:, :num_samples//time_step]

    return gen_demand, hh_junction_indexes, comm_junction_indexes, zero_junction_indexes, extreme_junction_indexes


if __name__ == '__main__':
    TIME_STEP = 1  # sampling rate in hours
    DURATION = 8760  # in hours
    YEARLY_PATTERN_NUM_HARMONICS = 4
    SUMMER_AMPLITUDE_RANGE = (2, 3)  # random lower and upper bounds to create a peak during summer
    SUMMER_ROLLING_RATE = 0.2
    SUMMER_START = 0.4166666666666667
    MIN_P_COMMERCIAL = 0.25
    MAX_P_COMMERCIAL = 0.35
    PROFILE_HOUSEHOLD = (0, 2, 1, 0)  # Consumption from 00.00-06.00, 06.00-12.00, 12.00-18.00, 18.00-00.00
    PROFILE_COMMERCIAL = (2, 2, 2, 1)  # each number represents low: 0, medium: 1 or high: 2 consumption
    PROFILE_EXTREME = (2, 2, 2, 2)
    MIN_NOISE = 0.02
    MAX_NOISE = 0.2
    ZERO_DEM_RATE = 0.1232
    EXTREME_DEM_RATE = 0.02
    MAX_EXTREME_DEM_JUNCTIONS = 2

    wn = wntr.network.WaterNetworkModel("../inputs/public/new_york.inp")

    start = time()
    for _ in range(5):
        demand, hh_keys, comm_keys, zero_demand_nodes, extreme_demand_nodes = generate_demand(
            wn=wn,
            time_step=TIME_STEP,
            duration=DURATION,
            yearly_pattern_num_harmonics=YEARLY_PATTERN_NUM_HARMONICS,
            summer_amplitude_range=SUMMER_AMPLITUDE_RANGE,
            summer_start=SUMMER_START,
            summer_rolling_rate=SUMMER_ROLLING_RATE,
            min_p_commercial=MIN_P_COMMERCIAL,
            max_p_commercial=MAX_P_COMMERCIAL,
            profile_household=PROFILE_HOUSEHOLD,
            profile_commercial=PROFILE_COMMERCIAL,
            profile_extreme=PROFILE_EXTREME,
            min_noise=MIN_NOISE,
            max_noise=MAX_NOISE,
            zero_dem_rate=ZERO_DEM_RATE,
            extreme_dem_rate=EXTREME_DEM_RATE,
            max_extreme_dem_junctions=MAX_EXTREME_DEM_JUNCTIONS
        )

        # print(f"hh_nodes: {hh_keys}")
        # print(f"comm_nodes: {comm_keys}")
        # print(f"zero_demand_nodes: {zero_demand_nodes}")
        # print(f"extreme_demand_nodes: {extreme_demand_nodes}")

    end = time()
    no_async_duration = end - start
    print(f'Execution time NO_async = {no_async_duration} sec')

    # plt.figure(figsize=(15, 4))
    # plt.plot(hh_demand[20], label="hh")
    # plt.plot(comm_demand[5], label="comm")
    # plt.plot(extreme_demand[0], label="extreme")
    # plt.plot(zero_demand[0], label="zero")
    #
    # plt.show()
    #
    # plt.plot(hh_demand[20, :168], label="hh")
    # plt.plot(comm_demand[5, :168], label="comm")
    # plt.plot(extreme_demand[0, :168], label="extreme")
    # plt.plot(zero_demand[0, :168], label="zero")
    # plt.show()

    samples_per_hour = int(1 / TIME_STEP) if TIME_STEP <= 1 else 1
    x_len = int(samples_per_hour * 168)
    plt.figure(figsize=(20, 2))
    idx = np.random.choice(hh_keys, 5, replace=False)

    plt.plot(range(x_len), demand[idx[0], :x_len], alpha=1)
    plt.plot(range(x_len), demand[idx[1], :x_len], alpha=1)
    plt.plot(range(x_len), demand[idx[2], :x_len], alpha=1)
    plt.show()

    # if x_len > 24:
    #     plt.figure(figsize=(20, 2))
    #     result = seasonal_decompose(demand[idx[0]], model='additive', period=24 * samples_per_hour)
    #     plt.plot(range(len(result.trend)), result.trend, alpha=1)
    #     result = seasonal_decompose(demand[idx[1]], model='additive', period=24 * samples_per_hour)
    #     plt.plot(range(len(result.trend)), result.trend, alpha=1)
    #     result = seasonal_decompose(demand[idx[2]], model='additive', period=24 * samples_per_hour)
    #     plt.plot(range(len(result.trend)), result.trend, alpha=1)
    #     plt.show()

    plt.figure(figsize=(20, 2))
    idx = np.random.choice(comm_keys, 5, replace=False)
    plt.plot(range(x_len), demand[idx[0], :x_len], alpha=1)
    plt.plot(range(x_len), demand[idx[1], :x_len], alpha=1)
    plt.plot(range(x_len), demand[idx[2], :x_len], alpha=1)
    plt.show()

    # if x_len > 24:
    #     plt.figure(figsize=(20, 2))
    #     result = seasonal_decompose(demand[idx[0]], model='additive', period=24 * samples_per_hour)
    #     plt.plot(range(len(result.trend)), result.trend, alpha=1)
    #     result = seasonal_decompose(demand[idx[1]], model='additive', period=24 * samples_per_hour)
    #     plt.plot(range(len(result.trend)), result.trend, alpha=1)
    #     result = seasonal_decompose(demand[idx[2]], model='additive', period=24 * samples_per_hour)
    #     plt.plot(range(len(result.trend)), result.trend, alpha=1)
    #     plt.show()

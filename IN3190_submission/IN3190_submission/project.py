from pathlib import Path

import cartopy.crs as ccrs
import h5py
import matplotlib.pyplot as plt
import numpy as np
from geopy.distance import great_circle
from scipy.signal import convolve as scipy_convolve


# Path to the folder containing the 201 station files.
DATA_FOLDER = Path(__file__).parent / "data"
OUTPUT_FOLDER = Path(__file__).parent / "output" / "pdf"
PICKS_FOLDER = Path(__file__).parent / "output" / "picks"
FILTERED_DATA_FILE = Path(__file__).parent / "filtered_signals.h5"
NUMBER_OF_SAMPLES = 720000
TONGA_LATITUDE = -20.550
TONGA_LONGITUDE = -175.385


def get_filters():
    """Return the three FIR filters provided with the project."""
    h1 = np.array([
        9.3102e-04, -1.2991e-18, -1.1771e-03, -8.9350e-04,
        1.1279e-03, 2.3259e-03, -3.0497e-18, -3.7419e-03,
        -2.8954e-03, 3.5886e-03, 7.1273e-03, -6.7002e-18,
        -1.0473e-02, -7.7679e-03, 9.2793e-03, 1.7882e-02,
        -1.0958e-17, -2.5342e-02, -1.8731e-02, 2.2575e-02,
        4.4596e-02, -1.4316e-17, -7.1659e-02, -6.0472e-02,
        9.2253e-02, 3.0157e-01, 3.9980e-01, 3.0157e-01,
        9.2253e-02, -6.0472e-02, -7.1659e-02, -1.4316e-17,
        4.4596e-02, 2.2575e-02, -1.8731e-02, -2.5342e-02,
        -1.0958e-17, 1.7882e-02, 9.2793e-03, -7.7679e-03,
        -1.0473e-02, -6.7002e-18, 7.1273e-03, 3.5886e-03,
        -2.8954e-03, -3.7419e-03, -3.0497e-18, 2.3259e-03,
        1.1279e-03, -8.9350e-04, -1.1771e-03, -1.2991e-18,
        9.3102e-04,
    ])

    h2 = np.array([
        6.8867e-04, -1.0409e-18, -8.7071e-04, -1.6144e-04,
        2.4454e-03, 4.3979e-03, 2.9653e-03, 1.8510e-04,
        1.9464e-03, 9.1274e-03, 1.2922e-02, 5.3683e-03,
        -6.4293e-03, -6.1213e-03, 7.3124e-03, 1.0978e-02,
        -1.3170e-02, -4.5946e-02, -4.7642e-02, -1.5176e-02,
        -2.2060e-03, -5.5677e-02, -1.3549e-01, -1.3111e-01,
        1.6668e-02, 2.2307e-01, 3.2035e-01, 2.2307e-01,
        1.6668e-02, -1.3111e-01, -1.3549e-01, -5.5677e-02,
        -2.2060e-03, -1.5176e-02, -4.7642e-02, -4.5946e-02,
        -1.3170e-02, 1.0978e-02, 7.3124e-03, -6.1213e-03,
        -6.4293e-03, 5.3683e-03, 1.2922e-02, 9.1274e-03,
        1.9464e-03, 1.8510e-04, 2.9653e-03, 4.3979e-03,
        2.4454e-03, -1.6144e-04, -8.7071e-04, -1.0409e-18,
        6.8867e-04,
    ])

    h3 = np.array([
        -2.4366e-04, -2.6135e-19, 3.0807e-04, 7.3294e-04,
        1.3147e-03, 2.0667e-03, 2.9630e-03, 3.9300e-03,
        4.8428e-03, 5.5289e-03, 5.7792e-03, 5.3643e-03,
        4.0571e-03, 1.6578e-03, -1.9803e-03, -6.9275e-03,
        -1.3160e-02, -2.0549e-02, -2.8859e-02, -3.7759e-02,
        -4.6838e-02, -5.5636e-02, -6.3672e-02, -7.0487e-02,
        -7.5676e-02, -7.8923e-02, 9.2033e-01, -7.8923e-02,
        -7.5676e-02, -7.0487e-02, -6.3672e-02, -5.5636e-02,
        -4.6838e-02, -3.7759e-02, -2.8859e-02, -2.0549e-02,
        -1.3160e-02, -6.9275e-03, -1.9803e-03, 1.6578e-03,
        4.0571e-03, 5.3643e-03, 5.7792e-03, 5.5289e-03,
        4.8428e-03, 3.9300e-03, 2.9630e-03, 2.0667e-03,
        1.3147e-03, 7.3294e-04, 3.0807e-04, -2.6135e-19,
        -2.4366e-04,
    ])

    return h1, h2, h3


def plot_impulse_responses(h1, h2, h3):
    """Plot the three provided FIR filter impulse responses."""
    filters = [h1, h2, h3]
    names = ["h1[n]", "h2[n]", "h3[n]"]

    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    for ax, h, name in zip(axs, filters, names):
        n = np.arange(len(h))
        ax.plot(n, h, "o", markersize=4)
        ax.set_ylabel("Amplitude")
        ax.set_title(name)

    axs[-1].set_xlabel("Sample n")
    fig.suptitle("FIR filter impulse responses")

    plt.tight_layout()
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FOLDER / "task2_impulse_responses.pdf", bbox_inches="tight")
    plt.close(fig)


def convolve_signal(x, h, ylen_choice):
    """Calculate the direct convolution of x and h."""
    x = np.asarray(x)
    h = np.asarray(h)

    signal_length = len(x)
    filter_length = len(h)
    full_length = signal_length + filter_length - 1
    y_full = np.zeros(full_length)

    for k in range(full_length):
        j_start = max(0, k - filter_length + 1)
        j_stop = min(k, signal_length - 1)

        for j in range(j_start, j_stop + 1):
            filter_index = k - j
            y_full[k] += x[j] * h[filter_index]

    if ylen_choice == 1:
        return y_full

    if ylen_choice == 0:
        start_index = (filter_length - 1) // 2
        stop_index = start_index + signal_length
        return y_full[start_index:stop_index]

    raise ValueError("ylen_choice must be 0 or 1")


def test_convolution():
    """Compare our convolution with SciPy using a small example."""
    x = np.array([1, 2, 3, 2, 1])
    h = np.array([1, -1, 0.5])

    own_full = convolve_signal(x, h, 1)
    scipy_full = scipy_convolve(x, h, mode="full")

    own_same = convolve_signal(x, h, 0)
    scipy_same = scipy_convolve(x, h, mode="same")

    print("\nTask 2b convolution test")
    print(f"x = {x}")
    print(f"h = {h}")
    print(f"Our full result:   {own_full}")
    print(f"SciPy full result: {scipy_full}")
    print(f"Full results match: {np.allclose(own_full, scipy_full)}")
    print(f"Our same result:   {own_same}")
    print(f"SciPy same result: {scipy_same}")
    print(f"Same results match: {np.allclose(own_same, scipy_same)}")


def calculate_dtft(h, N, fs):
    """Calculate the DTFT of h at N frequency points."""
    frequency = np.linspace(-fs / 2, fs / 2, N, endpoint=False)
    omega = 2 * np.pi * frequency / fs
    H = np.zeros(N, dtype=complex)

    for k in range(N):
        for n in range(len(h)):
            exponent = -1j * omega[k] * n
            H[k] += h[n] * np.exp(exponent)

    return H, frequency


def test_dtft():
    """Test the DTFT using h[n] = delta[n] + delta[n - 1]."""
    h = np.array([1, 1])
    N = 8
    fs = 8

    H, frequency = calculate_dtft(h, N, fs)
    omega = 2 * np.pi * frequency / fs
    expected_H = 1 + np.exp(-1j * omega)

    print("\nTask 2c DTFT test")
    print(f"h = {h}")
    print(f"Frequency axis: {frequency}")
    print(f"DTFT result is correct: {np.allclose(H, expected_H)}")


def filter_all_signals(file_list, h1, h2, h3, output_path):
    """Filter every station signal and store the results in one HDF5 file."""
    with h5py.File(output_path, "w") as output_file:
        for i in range(len(file_list)):
            file_path = file_list[i]
            signal, time_hours, sample_interval, start_time = load_signal(file_path)

            if len(signal) < NUMBER_OF_SAMPLES:
                missing_samples = NUMBER_OF_SAMPLES - len(signal)
                signal = np.pad(signal, (0, missing_samples))

            filtered_h1 = scipy_convolve(signal, h1, mode="same")
            filtered_h2 = scipy_convolve(signal, h2, mode="same")
            filtered_h3 = scipy_convolve(signal, h3, mode="same")

            station_group = output_file.create_group(file_path.stem)
            station_group.create_dataset("h1", data=filtered_h1)
            station_group.create_dataset("h2", data=filtered_h2)
            station_group.create_dataset("h3", data=filtered_h3)

            station_group.attrs["source_file"] = file_path.name
            station_group.attrs["sample_interval"] = sample_interval
            station_group.attrs["start_time"] = start_time

            print(f"Filtered station {i + 1} of {len(file_list)}: {file_path.name}")

    return output_path


def plot_frequency_responses(h1, h2, h3, N, fs):
    """Plot the positive-frequency response of the three FIR filters."""
    H1, frequency1 = calculate_dtft(h1, N, fs)
    H2, frequency2 = calculate_dtft(h2, N, fs)
    H3, frequency3 = calculate_dtft(h3, N, fs)

    magnitude1 = np.abs(H1)
    magnitude2 = np.abs(H2)
    magnitude3 = np.abs(H3)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(frequency1, magnitude1, label="h1")
    ax.plot(frequency2, magnitude2, label="h2")
    ax.plot(frequency3, magnitude3, label="h3")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title("FIR filter frequency responses")
    ax.set_xlim(0, fs / 2)
    ax.legend()

    plt.tight_layout()
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FOLDER / "task2_frequency_responses.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_station_test(file_path, h1, h2, h3):
    """Plot the raw and filtered signals for one station."""
    signal, time_hours, sample_interval, start_time = load_signal(file_path)

    if len(signal) < NUMBER_OF_SAMPLES:
        missing_samples = NUMBER_OF_SAMPLES - len(signal)
        signal = np.pad(signal, (0, missing_samples))
        time_hours = np.arange(NUMBER_OF_SAMPLES) * sample_interval / 3600

    filtered_h1 = scipy_convolve(signal, h1, mode="same")
    filtered_h2 = scipy_convolve(signal, h2, mode="same")
    filtered_h3 = scipy_convolve(signal, h3, mode="same")

    plot_step = 10
    fig, axs = plt.subplots(4, 1, figsize=(12, 9), sharex=True)

    axs[0].plot(time_hours[::plot_step], signal[::plot_step])
    axs[0].set_title("Raw signal")
    axs[0].set_ylabel("Amplitude")

    axs[1].plot(time_hours[::plot_step], filtered_h1[::plot_step])
    axs[1].set_title("Filtered with h1 (lowpass)")
    axs[1].set_ylabel("Amplitude")

    axs[2].plot(time_hours[::plot_step], filtered_h2[::plot_step])
    axs[2].set_title("Filtered with h2 (bandpass)")
    axs[2].set_ylabel("Amplitude")

    axs[3].plot(time_hours[::plot_step], filtered_h3[::plot_step])
    axs[3].set_title("Filtered with h3 (highpass)")
    axs[3].set_ylabel("Amplitude")
    axs[3].set_xlabel("Time after start (hours)")

    fig.suptitle(file_path.name)
    plt.tight_layout()

    print("\nClick on the start of the higher-frequency arrival")
    clicked_point = plt.ginput(1, timeout=-1)
    arrival_time = clicked_point[0][0]

    print(f"Selected arrival time: {arrival_time:.3f} hours after start")
    plt.close(fig)

    return arrival_time


def pick_arrival_times(file_list, distances, h1, h2, h3, start_index, stop_index, session_name):
    """Pick and save arrival times for stations sorted by distance."""
    sorted_indices = np.argsort(distances)
    station_indices = []
    arrival_times = []
    PICKS_FOLDER.mkdir(parents=True, exist_ok=True)
    output_file = PICKS_FOLDER / f"{session_name}.npz"

    if output_file.exists():
        with np.load(output_file) as saved_data:
            station_indices = list(saved_data["station_indices"])
            arrival_times = list(saved_data["arrival_times"])

        start_index = start_index + len(arrival_times)
        print(f"\nContinuing after {len(arrival_times)} saved picks")

    for i in range(start_index, stop_index + 1):
        station_index = sorted_indices[i]

        print(f"\nStation {i + 1} of {stop_index + 1}")
        print(f"Name: {file_list[station_index].name}")
        print(f"Distance: {distances[station_index]:.2f} km")

        arrival_time = plot_station_test(file_list[station_index], h1, h2, h3)

        station_indices.append(station_index)
        arrival_times.append(arrival_time)

        np.savez(output_file, station_indices=station_indices, arrival_times=arrival_times)
        print(f"Current picks saved in {output_file.name}")

    print(f"\nArrival picks saved in {output_file}")

    return station_indices, arrival_times


def analyze_arrival_times(distances, picks_file):
    """plot arrival picks and estimate the celerity"""
    saved_data = np.load(picks_file)
    station_indices = saved_data["station_indices"].astype(int)
    arrival_times = saved_data["arrival_times"]
    saved_data.close()

    # data starts at 04:00 and the eruption was at 04:14:45
    eruption_time = 14.75 / 60
    travel_hours = arrival_times - eruption_time
    travel_seconds = travel_hours * 3600

    distance_km = []
    for station_index in station_indices:
        distance_km.append(distances[station_index])

    distance_km = np.array(distance_km)
    distance_m = distance_km * 1000

    # find points which are far away from the main trend
    celerities = distance_m / travel_seconds
    median_celerity = np.median(celerities)
    expected_seconds = distance_m / median_celerity
    time_difference = (travel_seconds - expected_seconds) / 3600

    q1 = np.percentile(time_difference, 25)
    q3 = np.percentile(time_difference, 75)
    iqr = q3 - q1
    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr

    used_indices = []
    outlier_indices = []

    for i in range(len(time_difference)):
        if lower_limit <= time_difference[i] <= upper_limit:
            used_indices.append(i)
        else:
            outlier_indices.append(i)

    fit = np.polyfit(travel_seconds[used_indices], distance_m[used_indices], 1)
    estimated_celerity = fit[0]
    intercept = fit[1]

    line_time = np.linspace(min(travel_hours[used_indices]), max(travel_hours[used_indices]), 100)
    line_distance = (estimated_celerity * line_time * 3600 + intercept) / 1000

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(travel_hours[used_indices], distance_km[used_indices], s=18, label="used picks")
    ax.scatter(travel_hours[outlier_indices], distance_km[outlier_indices], s=28, color="red", marker="x", label="outliers")
    ax.plot(line_time, line_distance, color="black", label=f"fit: {estimated_celerity:.1f} m/s")

    ax.set_xlabel("travel time (hours)")
    ax.set_ylabel("distance from Hunga Tonga (km)")
    ax.set_title("infrasound arrival times and celerity estimate")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FOLDER / "task3_celerity.pdf", bbox_inches="tight")
    plt.close(fig)

    print("\ntask 3b celerity analysis")
    print(f"number of used picks: {len(used_indices)}")
    print(f"number of outliers: {len(outlier_indices)}")
    print(f"estimated celerity: {estimated_celerity:.2f} m/s")

    return estimated_celerity, used_indices, outlier_indices


def load_metadata(file_list):
    """Load station names and coordinates from all HDF5 files."""
    station_names = []
    latitudes = []
    longitudes = []

    for file_path in file_list:
        station_name = file_path.name

        with h5py.File(file_path, "r") as h5_file:
            latitude = h5_file.attrs["latitude"]
            longitude = h5_file.attrs["longitude"]

        station_names.append(station_name)
        latitudes.append(latitude)
        longitudes.append(longitude)

    return station_names, latitudes, longitudes


def load_signal(file_path):
    """Load the pressure signal and time information from one HDF5 file."""
    with h5py.File(file_path, "r") as h5_file:
        group_name = list(h5_file.keys())[0]
        group = h5_file[group_name]
        dataset_name = list(group.keys())[0]
        dataset = group[dataset_name]

        signal = dataset[:]
        sample_interval = dataset.attrs["delta"]
        start_time = dataset.attrs["starttime"]

    time_hours = np.arange(len(signal)) * sample_interval / 3600

    return signal, time_hours, sample_interval, start_time


def plot_example_signal(time_hours, signal, station_name):
    """Plot one example signal from the dataset."""
    plot_step = 10

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        time_hours[::plot_step],
        signal[::plot_step],
        linewidth=0.7,
    )

    ax.set_xlabel("Time after start (hours)")
    ax.set_ylabel("Pressure signal")
    ax.set_title(f"Raw pressure signal from {station_name}")
    ax.grid(True)

    plt.tight_layout()
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FOLDER / "task1_example_signal.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_map(latitudes, longitudes):
    """Plot the stations and Hunga Tonga on a world map."""
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.Robinson())

    ax.set_global()
    ax.coastlines()
    ax.gridlines()

    # Plot stations.
    ax.scatter(
        longitudes,
        latitudes,
        marker="^",
        facecolors="none",
        edgecolors="blue",
        transform=ccrs.PlateCarree(),
        label="Stations",
    )

    # Plot Hunga Tonga.
    ax.scatter(
        TONGA_LONGITUDE,
        TONGA_LATITUDE,
        marker="x",
        color="red",
        s=80,
        transform=ccrs.PlateCarree(),
        label="Hunga Tonga",
    )

    ax.set_title("Hunga Tonga and measurement stations")
    ax.legend(loc="lower left")
    plt.tight_layout()
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FOLDER / "task1_map.pdf", bbox_inches="tight")
    plt.close(fig)


def calculate_distances(latitudes, longitudes):
    """Calculate the great circle distance to each station in km."""
    distances = []
    tonga_position = (TONGA_LATITUDE, TONGA_LONGITUDE)

    for latitude, longitude in zip(latitudes, longitudes):
        station_position = (latitude, longitude)
        distance = great_circle(tonga_position, station_position).km
        distances.append(distance)

    return distances


def plot_sorted_distances(distances):
    """Plot all distances sorted from shortest to longest."""
    sorted_distances = sorted(distances)
    station_numbers = range(1, len(sorted_distances) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(station_numbers, sorted_distances, s=15)

    ax.set_xlabel("Station number sorted by distance")
    ax.set_ylabel("Distance from Hunga Tonga (km)")
    ax.set_title("Sorted great circle distances")
    ax.grid(True)

    plt.tight_layout()
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FOLDER / "task1_sorted_distances.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    file_list = sorted(DATA_FOLDER.glob("*.h5"))
    print(f"Found {len(file_list)} HDF5 files.\n")

    if not file_list:
        raise FileNotFoundError(f"No HDF5 files found in {DATA_FOLDER}")

    # Task 1a - Read metadata and one example signal.
    station_names, latitudes, longitudes = load_metadata(file_list)
    signal, time_hours, sample_interval, start_time = load_signal(file_list[0])

    # Print the first three stations to check the result.
    for i in range(3):
        print(
            f"{station_names[i]}: "
            f"latitude = {latitudes[i]}, longitude = {longitudes[i]}"
        )

    print(f"\nExample signal: {station_names[0]}")
    print(f"Number of samples: {len(signal)}")
    print(f"Sample interval: {sample_interval} s")
    print(f"Sampling frequency: {1 / sample_interval:.1f} Hz")
    print(f"Start time: {start_time}")

    # Task 1b - Calculate the distance from Hunga Tonga to each station.
    distances = calculate_distances(latitudes, longitudes)

    nearest_index = distances.index(min(distances))
    furthest_index = distances.index(max(distances))

    print(
        f"\nNearest station: {station_names[nearest_index]}, "
        f"distance = {distances[nearest_index]:.2f} km"
    )
    print(
        f"Furthest station: {station_names[furthest_index]}, "
        f"distance = {distances[furthest_index]:.2f} km"
    )

    # Task 1a figures.
    plot_example_signal(time_hours, signal, station_names[0])
    plot_map(latitudes, longitudes)

    # Task 1c - Plot all sorted distances.
    plot_sorted_distances(distances)

    # Task 2a - Plot the three provided FIR filters.
    h1, h2, h3 = get_filters()
    plot_impulse_responses(h1, h2, h3)

    # Task 2b - Verify our direct convolution against SciPy.
    test_convolution()

    # Task 2c - Verify our direct DTFT calculation.
    test_dtft()

    # Task 2d - Plot the positive-frequency response of all filters.
    N = 1000
    fs = 1 / sample_interval
    plot_frequency_responses(h1, h2, h3, N, fs)

    # Task 2e - Identify the three filter types.
    print("\nTask 2e filter types")
    print("h1 is a lowpass filter")
    print("h2 is a bandpass filter")
    print("h3 is a highpass filter")

    # Task 2f - Filter all station signals and store the results.
    if FILTERED_DATA_FILE.exists():
        print(f"\nFiltered data already exists: {FILTERED_DATA_FILE.name}")
    else:
        filter_all_signals(file_list, h1, h2, h3, FILTERED_DATA_FILE)

    # Task 3a - Pick arrival times for all stations, sorted by distance.
    pick_arrival_times(file_list, distances, h1, h2, h3, 0, len(file_list) - 1, "all_stations")

    # Task 3b - Plot the picks, remove outliers, and estimate celerity.
    analyze_arrival_times(distances, PICKS_FOLDER / "all_stations.npz")

if __name__ == "__main__":
    main()

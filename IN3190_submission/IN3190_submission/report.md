# Answers and notes

These are the answers and results I got while working through the project. I
will use this as the basis for the final report.

## Task 1 - The overview

### Task 1a

I started by reading the station name and the latitude and longitude from all
201 HDF5 files. I also loaded one of the pressure signals just to check how the
data was stored and that I had read it correctly. The example signal had
720000 samples. The time between each sample was 0.1 seconds, which gives a
sampling frequency of 10 Hz.

[Figure 1: Example of one raw pressure signal](output/pdf/task1_example_signal.pdf)

After reading the coordinates I plotted all the stations on a map together
with the location of Hunga Tonga. This made it easier to see how spread out the
stations actually are.

[Figure 2: Hunga Tonga and the measurement stations](output/pdf/task1_map.pdf)

### Task 1b

I calculated the great circle distance from Hunga Tonga to every station. The
shortest distance I found was 2137.71 km, for station
`AM.RF356_unfilt.h5`. The longest one was 17687.13 km, for station
`AM.R10D3_unfilt.h5`.

### Task 1c

I sorted all 201 distances from shortest to longest and plotted them. The
result is shown in Figure 3.

[Figure 3: Sorted great circle distances](output/pdf/task1_sorted_distances.pdf)

## Task 2 - Data filtering

### Task 2a

I first plotted the three FIR filters that were provided in the assignment.
All three impulse responses have 53 samples, but their shapes are quite
different. They are shown together in Figure 4.

[Figure 4: The three FIR filter impulse responses](output/pdf/task2_impulse_responses.pdf)

### Task 2b

For the convolution function I used two nested loops. I did it this way since
it follows the convolution sum quite directly and I found it easier to see
which samples were being multiplied. The function can either return the full
result with length M + N - 1, or only the middle part with the same length as
the input signal.

I tested both versions on a small example and compared the answers with
SciPy's convolution function. The results were the same in both cases.

### Task 2c

I calculated the DTFT directly from the sum in the assignment, again using
nested loops. The function returns both the complex DTFT values and a frequency
axis in Hz. To check it, I used the small signal h[n] = delta[n] + delta[n - 1]
and compared it with the DTFT I expected for that signal. The test matched.

### Task 2d

I plotted the absolute value of the frequency response for h1, h2, and h3 in
the same figure. I only displayed the positive frequencies from 0 Hz to the
Nyquist frequency, which is 5 Hz for these signals. The result is shown in
Figure 5.

[Figure 5: Frequency responses of h1, h2, and h3](output/pdf/task2_frequency_responses.pdf)

### Task 2e

From the frequency response plots I got the following filter types:

- h1 is a lowpass filter. It keeps the low frequencies and reduces the higher
  frequencies.
- h2 is a bandpass filter. It mainly keeps a limited range in the middle.
- h3 is a highpass filter. It removes the lower frequencies and keeps the
  higher frequencies.

### Task 2f

I filtered all 201 pressure signals with h1, h2, and h3. For this part I used
SciPy's convolution with `mode="same"`, since using my own nested loop function
on signals this long would take much more time. I processed one station at a
time instead of keeping all the filtered signals in memory at once.

The results were stored in `filtered_signals.h5`. Each station has its own
group in the file, with datasets called `h1`, `h2`, and `h3`. Three of the
original signals were a little shorter than 720000 samples, so I filled the
remaining part with zeros in the same way as in the provided example.

### Task 2g

I tried looking at how the section plot could be made, but I did not manage to
finish this part.

## Task 3 - Celerity estimation

### Task 3a

For this part I went through all 201 stations manually, ordered from the
shortest distance to the longest distance. For each station the program showed
the raw signal and the three filtered signals. I mostly looked at the h2
bandpass result because the arrival was normally easiest to see there. I also
checked h3 when I was unsure.

I clicked close to the beginning of the first clear higher-frequency wave
packet instead of clicking on its largest peak. Some of the stations were very
noisy, so a few of the choices were difficult and were more uncertain than the
others. All arrival times and station indices were saved in
`output/picks/all_stations.npz`.

### Task 3b

The times I clicked were measured from the beginning of the data at 04:00:00
UTC. The main eruption was at 04:14:45 UTC, so I subtracted these 14 minutes
and 45 seconds to get the actual travel times. I then plotted travel time on
the x axis and the great circle distance on the y axis.

Because some of my manual picks were uncertain, there were a few points that
were clearly far away from the main group. I first used the median celerity to
get a rough main trend. After that I looked at the time difference from this
trend and used the 1.5 times IQR rule to separate the clearest outliers. This
marked 31 points as outliers, while 170 points were kept for the line fit. The
outliers are still displayed in the figure as red crosses, so they are not just
removed without being shown.

[Figure 6: Arrival-time picks, outliers, and fitted celerity](output/pdf/task3_celerity.pdf)

The fitted line gave a celerity of 284.92 m/s. This is in the range I expected
for globally propagating infrasound, and it is fairly close to the value of
about 300 m/s used by Vergoz et al. (2022).

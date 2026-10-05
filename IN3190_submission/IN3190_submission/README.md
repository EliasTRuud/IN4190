# IN3190 mandatory project

The main program is `project.py`. The report figures are stored in
`output/pdf`, and the manually picked arrival times for Task 3 are stored in
`output/picks/all_stations.npz`.

The official collection of 201 HDF5 station files is not included because it
is supplied separately by the course and is too large for the submission. To
run the complete program, place those files in a folder called `data` next to
`project.py`.

Install the required packages with:

```text
python -m pip install -r requirements.txt
```

Run the program with:

```text
python project.py
```

Task 2g was not completed. The answers and numerical results are collected in
`report.md`.

## Use of GPT

I used GPT as a programming aid while working on this project. It was especially
helpful for Task 3, which I struggled quite a bit with. GPT helped me set up the
interactive program where the arrival time is selected by clicking on each
signal plot. I still went through all 201 stations myself and made the actual
arrival-time picks manually.

GPT also helped me organize the project files and clean up the code after I had
worked through the different tasks. This included putting the generated figures
and saved results in suitable folders and making the code more consistent and
easier to run. GPT also helped me write and organize this README file.

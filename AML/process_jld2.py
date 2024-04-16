from juliacall import Main as jl
import numpy as np
jl.seval("using JLD2")

def get_data(data_filename, problemtype="results1", N_freq=2):
    data = jl.load(data_filename)
    relevant_entries = [entry for entry in data[problemtype] if jl.size(entry["om"], 2) == N_freq]

    return relevant_entries

def get_x_y(relevant_entries):
    x_list = []
    for entry in relevant_entries:
        x1_x2_x12 = np.array(entry["x1_x2_x12"], dtype=np.float32)
        om = np.array(entry["om"][:,1:], dtype=np.float32).reshape(-1) # Omit the 'zero' frequency entries
        x_entry = np.concatenate([x1_x2_x12, om])
        x_list.append(x_entry)

    y_list = [entry["objHist"][-1] for entry in relevant_entries]

    x = np.array(x_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    return x, y




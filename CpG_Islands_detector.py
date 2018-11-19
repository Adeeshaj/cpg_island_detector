#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 17:59:53 2018

@author: adeeshaj
"""

import numpy as np

def get_input_sequence(filename):
    file = open(filename, "r")
    sequence = file.read().strip()
            
    out = []
    
    for char in sequence:
        if char == "a":
            out.append(0)
        elif char == "c":
            out.append(1)
        elif char == "g":
            out.append(2)
        else:
            out.append(3)
            
    return np.array(out)


#the Viterbi algorithm
def viterbi(emission_probs, transition_probs, initial_dist, emissions):
    probs = emission_probs[:, 0] * initial_dist
    stack = []
    num_states = transition_probs.shape[0]

    for emission in emissions[1:]:
        trans_probs = transition_probs * np.row_stack(probs)
        max_col_ixs = np.argmax(trans_probs, axis=0)
        probs = emission_probs[:, emission] * trans_probs[max_col_ixs, np.arange(num_states)]
        probs = [x*4.21 for x in probs] #to prevent values going to be 0
        stack.append(max_col_ixs)

    state_seq = [np.argmax(probs)]

    while stack:
        max_col_ixs = stack.pop()
        state_seq.append(max_col_ixs[state_seq[-1]])

    state_seq.reverse()

    return state_seq

states = ["O","A+", "C+", "G+", "T+", "A-", "C-", "G-", "T-"]
n_states = len(states)

observations = ["a", "c", "g", "t"]
n_observations = len(observations)

start_probability = np.array([0,0.0725193,0.1637630,0.1788242,0.0754545,0.1322050,0.1267006,0.1226380,0.1278950])

transition_probability = np.array([
  [0,0.0725193,0.1637630,0.1788242,0.0754545,0.1322050,0.1267006,0.1226380,0.1278950],
  [0.0010000,0.1762237,0.2682517,0.4170629,0.1174825,0.0035964,0.0054745,0.0085104,0.0023976],
  [0.0010000,0.1672435,0.3599201,0.2679840,0.1838722,0.0034131,0.0073453,0.0054690,0.0037524],
  [0.0010000,0.1576223,0.3318881,0.3671328,0.1223776,0.0032167,0.0067732,0.0074915,0.0024975],
  [0.0010000,0.0773426,0.3475514,0.3759440,0.1781818,0.0015784,0.0070929,0.0076723,0.0036363],
  [0.0010000,0.0002997,0.0002047,0.0002837,0.0002097,0.2994045,0.2045904,0.2844305,0.2095804],
  [0.0010000,0.0003216,0.0002977,0.0000769,0.0003016,0.3213566,0.2974045,0.0778441,0.3013966],
  [0.0010000,0.0001768,0.0002387,0.0002917,0.0002917,0.1766463,0.2385224,0.2914165,0.2914155],
  [0.0010000,0.0002477,0.0002457,0.0002977,0.0002077,0.2475044,0.2455084,0.2974035,0.2075844]])

emission_probability = np.array([
  [0,0,0,0],
  [1,0,0,0],
  [0,1,0,0],
  [0,0,1,0],
  [0,0,0,1],
  [1,0,0,0],
  [0,1,0,0],
  [0,0,1,0],
  [0,0,0,1]
])

input_sequence = get_input_sequence("input.txt")

output = viterbi( emission_probability,transition_probability,start_probability, input_sequence)

file = open("output.txt","w")
for x in output:
    file.write(states[x][-1])
file.close()




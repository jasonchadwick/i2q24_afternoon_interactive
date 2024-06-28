import numpy as np
from stim import Tableau
import os
import subprocess
import signal
import time
import random
from contextlib import contextmanager
from collections import deque

from qiskit.circuit.library import TGate, RZGate, HGate
from qiskit import QuantumCircuit

def get_gridsynth_decomposition(theta, precision_decimal_count=10, seed=0):
    '''
    Args:
    theta_factor: The angle whose decomposition we want 
    precision_decimal_count: The value of epsilon is e^-(precision_decimal_count)

    Returns:
    The Clifford+T decomposition for rz(theta) 
    '''
    if theta>=0:
        string=subprocess.run(["gridsynth", str(theta), "-d "+str(precision_decimal_count), "-r "+str(seed)], capture_output=True, text=True).stdout
    else:
        string=subprocess.run(["gridsynth", "(-" + str(abs(theta)) + ")", "-d "+str(precision_decimal_count), "-r "+str(seed)], capture_output=True, text=True).stdout
    return string

def compute_elements(angles):
    '''
    Args:
    angles: list of angles that are parameters of the u gate
    '''
    theta, phi, lambda_ =angles[0], angles[1], angles[2]
    a00=np.cos(theta/2)
    a01=-np.exp(1j*lambda_)*np.sin(theta/2)
    a10=np.exp(1j*phi)*np.sin(theta/2)
    a11=np.exp(1j*(phi+lambda_))*np.cos(theta/2)

    return np.array([[a00, a01], [a10, a11]])

def is_rz_angle_Clifford(rz_angle):
    '''
    Args:
    rz_angle: Angle which is the argument of an rz gate
    '''
    rz_angle=rz_angle%(2*np.pi)
    if np.isclose(rz_angle, 0, atol=1e-8) or np.isclose(rz_angle, np.pi/2, atol=1e-8) or np.isclose(rz_angle, np.pi, atol=1e-8) or np.isclose(rz_angle, 3*np.pi/2, atol=1e-8) or np.isclose(rz_angle, 2*np.pi, atol=1e-3):
        return True
    else:
        print('Angle ', rz_angle, ' is NOT Clifford')
        return False

def is_Clifford(matrix):
    '''
    Args:
    matrix: 2x2 numpy array representing a single qubit gate
    '''
    try:
        Tableau.from_unitary_matrix(matrix, endian='little')
        return True
    except:
        return False

def rewrite_circuit_in_cliff_plus_T(qc:QuantumCircuit, precision_decimal_count=10):
    '''
    Args:
    qc: Quantum Circuit that has gates that are either single qubit 'u' gates or 'cx' gates

    Returns:
    qc_new: Quantum Circuit that has clifford+T gates
    '''
    qc_new=QuantumCircuit()
    for qreg in qc.qregs:
        qc_new.add_register(qreg)
    for creg in qc.cregs:
        qc_new.add_register(creg)
    
    gate_index=0
    for instr, qarg, carg in qc.data:
        print('Start gate number: ', gate_index)
        gate_index+=1
        
        if instr.name=='u':
            assert len(qarg)==1
            assert len(carg)==0
            if is_Clifford(compute_elements(instr.params)):
                continue # assuming a two-wide surface code which performs all single qubit Clifford gates in software
            else: # add only the T gates from the gridsynth decomposition -- skip the single qubit Cliffords since they are virtualized
                #find a rz-rx-rz decomposition of the u gate and then do a Clifford+T decomposition using gridsynth
                params=instr.params
                theta, phi, lambda_=params[0], params[1], params[2]
                alpha, beta, gamma=phi+np.pi/2, theta, lambda_-np.pi/2 # get the rz-h-rz-h-rz decomposition angles
                #print(alpha, beta, gamma)

                # get the gridsynthe decomposition for these angles
                alpha_string=get_gridsynth_decomposition(alpha, precision_decimal_count=precision_decimal_count)
                beta_string=get_gridsynth_decomposition(beta, precision_decimal_count=precision_decimal_count)
                gamma_string=get_gridsynth_decomposition(gamma, precision_decimal_count=precision_decimal_count)

                # add all the T gates from the gridsynth decomposition to the transpiled circuit ###
                t_instr=TGate()

                # for angle alpha
                if is_rz_angle_Clifford(alpha):
                    pass
                else:
                    for el in alpha_string:
                        if el=='T':
                            qc_new.append(t_instr, qarg, carg)
                
                # for angle beta
                if is_rz_angle_Clifford(beta):
                    pass
                else:
                    for el in beta_string:
                        if el=='T':
                            qc_new.append(t_instr, qarg, carg)
                
                # for angle gamma
                if is_rz_angle_Clifford(gamma):
                    pass
                else:
                    for el in gamma_string:
                        if el=='T':
                            qc_new.append(t_instr, qarg, carg) 
        else:
            qc_new.append(instr, qarg, carg)
    
    return qc_new

def bfs_shortest_path(graph, start, end, invalid_nodes=[]):
    '''
    Args:
    graph: Graph as an adjacency matrix of dimension VxV
    start: Start vertex
    end: End vertex
    invalid_nodes: List of nodes which are occupied and cannot be used for routing - union of magic state locations and qubit locations
    '''
    n = len(graph)
    visited = [False] * n
    parent = [-1] * n
    
    queue = deque()
    queue.append(start)
    visited[start] = True

    # Mark invalid nodes as visited
    # except the actual ending path
    for node in invalid_nodes:
        if node != end:
            visited[node] = True

    while queue:
        current = queue.popleft()
        if current == end:
            break
        
        for neighbor in range(n):
            if graph[current][neighbor] != 0 and not visited[neighbor]:
                queue.append(neighbor)
                visited[neighbor] = True
                parent[neighbor] = current
    
    # Reconstruct the shortest path from start to end
    shortest_path = []
    while end != -1:
        shortest_path.append(end)
        end = parent[end]
    shortest_path.reverse()
    
    return shortest_path

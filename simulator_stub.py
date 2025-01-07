'''
==================================================
File: simulator_stub.py
Project: simopty
File Created: Tuesday, 10th December 2024
Author: Edoardo Cabiati (edoardo.cabiati@mail.polimi.it)
Under the supervision of: Politecnico di Milano
==================================================
'''


"""
The simulator_stub.py module contains the classes used to interface with the simulator estimate latency and energy consumption
for a given mapping of the tasks onto the NoC grid.
In particular, to perform a simulation, the simulator requires a configuration (.JSON) file, which contains:
- the architectural parameters of the NoC grid (e.g., the number of PEs, the topology of the grid, the communication latency, etc.);
- the workload and corresponding mapping of the tasks onto the NoC grid.
The first is predefined and kept fixed, while the second can be synthsized by the mapper.py module, given the specific mapping.
that has been considered.
The Stub class should simply provide and interface from the main.py module to the simulator.
"""

import os
import sys
import platform
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import time

# print python version
PATH_TO_SIMULATOR = os.path.join("/Users/edoardocabiati/Desktop/Cose_brutte_PoliMI/_tesi/restart", "lib")
sys.path.append(PATH_TO_SIMULATOR)
import nocsim

class SimulatorStub:

    def __init__(self, path_to_executable = None):
        self.path_to_executable = path_to_executable

    def run_simulation(self, path_to_config_file, verbose=False):
        """
        Runs the simulation with the given configuration file.

        Parameters
        ----------
        path_to_config_file : str
            The path to the configuration file.

        Returns
        -------
        None
        """
        
        if verbose:
            print("Running simulation with configuration file: " + path_to_config_file)
        #os.system(self.path_to_executable + " " + path_to_config_file)
        start = time.time()
        results = nocsim.simulate(path_to_config_file, "")
        end = time.time()
        if verbose:
            print(f"Simulation completed in {end - start:.2f} seconds.")
        return results

    def run_simulation_on_processor(self, path_to_config_file, processor, verbose=False):
        """
        Runs the simulation with the given configuration file on the specified processor.

        Parameters
        ----------
        path_to_config_file : str
            The path to the configuration file.
        processor : int
            The processor on which the simulation is to be run.

        Returns
        -------
        None
        """
        
        if verbose:
            print(f"Running simulation on processor {processor} with configuration file: {path_to_config_file}")

        # Verifica se il sistema operativo è diverso da Linux
        if platform.system().lower() != "linux":
            if verbose:
                print("Setting processor affinity is not supported on this OS.")
        else:
            def set_affinity():
                os.system(f"taskset -p -c {processor} {os.getpid()}")
            p = multiprocessing.Process(target=set_affinity)
            p.start()
            p.join()
        start = time.time()
        # os.system(self.path_to_executable + " " + path_to_config_file)
        results = nocsim.simulate(path_to_config_file, "")
        end = time.time()
        if verbose:
            print(f"Simulation completed in {end - start:.2f} seconds.")

        return results

    def run_simulations_in_parallel(self, config_files, processors, verbose=False):
        """
        Runs simulations in parallel on different processors.

        Parameters
        ----------
        config_files : list of str
            List of paths to configuration files.
        processors : list of str
            List of processors to run the simulations on.
        verbose : bool
            If True, prints verbose output.

        Returns
        -------
        None
        """
        if verbose:
            print("Running batch of simulations in parallel.")
        start = time.time()
        if processors is None:
            processors = list(range(os.cpu_count()))
        with ThreadPoolExecutor(max_workers=len(processors)) as executor:
            futures = []
            for config_file, processor in zip(config_files, processors):
                futures.append(executor.submit(self.run_simulation_on_processor, config_file, processor, False))
            results = [future.result() for future in futures]  # Wait for all futures to complete
        end = time.time()
        if verbose:
            print(f"Simulations completed in {end - start:.2f} seconds.")
        
        return results
                



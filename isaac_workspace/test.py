# main.py
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": True})  # or False if you want GUI

print("Isaac Sim started successfully.")

simulation_app.close()

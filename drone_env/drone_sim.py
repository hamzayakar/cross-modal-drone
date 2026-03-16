import time
import pybullet as p
import pybullet_data

# Constants
MAX_SIMULATION_STEPS = 1000
TIME_STEP = 1.0 / 240.0

def test_wslg_visualizer():
    """
    Initializes the PyBullet GUI to test WSL2 graphical output.
    Loads a ground plane and drops a test robot with gravity.
    """
    # Connect to the PyBullet GUI
    physics_client = p.connect(p.GUI)
    
    # Add search path for default 3D models
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Set standard Earth gravity
    p.setGravity(0, 0, -9.81)
    
    # Load the ground plane
    plane_id = p.loadURDF("plane.urdf")
    
    # Drop position: [X=0, Y=0, Z=2 meters high]
    start_pos = [0, 0, 2]
    start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    
    # Load a test robot (R2D2) to stand in for our drone today
    robot_id = p.loadURDF("r2d2.urdf", start_pos, start_orientation)
    
    print("Simulation started! Look for a new 3D window on your taskbar.")
    
    # Step the simulation forward infinitely
    while True:
        p.stepSimulation()
        time.sleep(TIME_STEP)
        
    p.disconnect()

if __name__ == "__main__":
    test_wslg_visualizer()
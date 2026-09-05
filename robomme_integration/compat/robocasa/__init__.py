"""Import-only RoboCasa surface for isolated RoboMME policy training.

The shared OpenPI fork imports RoboCasa dataset symbols at module import time even when its
dedicated RoboMME LeRobot adapter is selected.  SageMaker RoboMME training deliberately does not
install the simulator.  These modules satisfy only that eager import surface; every dataset class
fails closed if it is instantiated.
"""

ROBOMME_IMPORT_ONLY = True

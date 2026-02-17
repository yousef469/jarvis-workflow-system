"""
NASA Module 3: Simulation Engine
===============================
Handles 6-DOF (Six Degrees of Freedom) flight dynamics.
Environment specific: Earth vs Mars.
"""

import numpy as np
import time
import json
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class Environment:
    name: str
    gravity: float  # m/s^2
    air_density: float # kg/m^3 at sea level
    wind: np.ndarray # (x, y, z) m/s
    temperature: float # Celsius

ENVIRONMENTS = {
    "EARTH": Environment("Earth", 9.806, 1.225, np.array([0, 0, 0]), 15.0),
    "MARS": Environment("Mars", 3.71, 0.015, np.array([2.0, 0, 0]), -60.0)
}

class FlightSimulator6DOF:
    def __init__(self, physics_config: Dict):
        self.config = physics_config
        self.mass = physics_config.get("total_mass_kg", 1000.0)
        self.com = np.array(physics_config.get("center_of_mass", [0, 0, 0]))
        
        # State Vectors
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.acceleration = np.array([0.0, 0.0, 0.0])
        
        # Orientation (Euler angles for simplicity)
        self.rotation = np.array([0.0, 0.0, 0.0])
        self.angular_velocity = np.array([0.0, 0.0, 0.0])
        
        self.env = ENVIRONMENTS["EARTH"]
        self.thrust_level = 0.0 # 0 to 1.0
        self.last_update = time.time()

    def set_environment(self, env_name: str):
        if env_name in ENVIRONMENTS:
            self.env = ENVIRONMENTS[env_name]

    def apply_physics_step(self, dt: float):
        """Standard NASA Physics Step (Module 3)"""
        
        # 1. Calculate Forces
        # Weight (Gravity)
        grav_force = np.array([0, -self.env.gravity * self.mass, 0])
        
        # Thrust (Simplified: mostly up/down for now)
        # Assuming max thrust is 2x Earth weight for a rocket
        max_thrust = 2 * 9.8 * self.mass
        thrust_force = np.array([0, self.thrust_level * max_thrust, 0])
        
        # Drag (Projected area simplified)
        # Fd = 1/2 * rho * v^2 * Cd * A
        rho = self.env.air_density
        v_mag = np.linalg.norm(self.velocity)
        drag_coeff = 0.5
        area = 10.0 # m^2
        drag_mag = 0.5 * rho * v_mag**2 * drag_coeff * area
        drag_force = -self.velocity * (drag_mag / (v_mag + 1e-6))
        
        # Net Force
        total_force = grav_force + thrust_force + drag_force
        
        # 2. Integration (Euler for now, RK4 preferred for production)
        self.acceleration = total_force / self.mass
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt
        
        # Ground collision
        if self.position[1] < 0:
            self.position[1] = 0
            self.velocity[1] = max(0, self.velocity[1]) # Stop falling
            self.velocity *= 0.8 # Friction

    def update(self) -> Dict:
        """Run a single real-time update"""
        now = time.time()
        dt = min(now - self.last_update, 0.1) # Limit dt to prevent instability
        self.last_update = now
        
        self.apply_physics_step(dt)
        
        return {
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "acceleration": self.acceleration.tolist(),
            "thrust": float(self.thrust_level),
            "env": self.env.name
        }

class ControlBrain:
    """
    NASA Module 4: Autonomous Flight Software (JARVIS)
    Uses PID control to maintain altitude or land.
    """
    def __init__(self, p=0.1, i=0.01, d=0.05):
        self.Kp = p
        self.Ki = i
        self.Kd = d
        
        self.integral = 0.0
        self.last_error = 0.0
        self.target_altitude = 100.0
        
    def compute_thrust(self, current_alt: float, current_vel: float, dt: float) -> float:
        """Computes thrust output (0 to 1.0) to reach target_altitude"""
        error = self.target_altitude - current_alt
        self.integral += error * dt
        derivative = (error - self.last_error) / dt if dt > 0 else 0
        
        # PID Output
        output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)
        
        # Add gravity compensation (Feed-forward)
        # 0.5 thrust is roughly hover in our 2x Earth weight engine simplification
        hover_thrust = 0.5 
        output += hover_thrust
        
        self.last_error = error
        return np.clip(output, 0.0, 1.0)

if __name__ == "__main__":
    # Test integrated Sim + Control
    sim = FlightSimulator6DOF({"total_mass_kg": 5000})
    brain = ControlBrain()
    brain.target_altitude = 50.0
    
    print(f"Starting JARVIS Controlled Flight Test (Target: {brain.target_altitude}m)")
    for i in range(20):
        data = sim.update()
        dt = 0.1
        sim.thrust_level = brain.compute_thrust(data["position"][1], data["velocity"][1], dt)
        
        print(f"T+{i*dt:.1f}s | Alt: {data['position'][1]:.2f}m | Thrust: {sim.thrust_level:.2f}")
        time.sleep(dt)

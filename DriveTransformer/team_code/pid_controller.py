import numpy as np

class DecouplePIDController(object):
    """
    Decoupled PID controller in steering and speed. Control steering with predicted trajectories of fixed distance and control speed (throttle and brake) with that of fixed time.
    """
    def __init__(self, speed_k_p=1.0, speed_k_i=0.0, speed_k_d=0.0, speed_n=20, steer_k_p=1.0, steer_k_i=0.0, steer_k_d=0.0, steer_n=6, brake_threshold=0.5, brake_k=1.0, brake_b=0.5):
        self.speed_k_p = speed_k_p # PID parameters for speed control
        self.speed_k_i = speed_k_i
        self.speed_k_d = speed_k_d
        self.speed_n = speed_n
        self.steer_k_p = steer_k_p # PID parameters for steer control
        self.steer_k_i = steer_k_i
        self.steer_k_d = steer_k_d
        self.steer_n = steer_n    
        self.speed_window = [] 
        self.steer_window = []
        self.brake_threshold = brake_threshold # brake if desired_speed is smaller than the number
        self.brake_k = brake_k # brake = brake_b + self.brake_k * (current_speed - desired_speed)
        self.brake_b = brake_b
    
    def step(self, traj_fix_time, traj_fix_dist, current_speed):
        
        desired_speed = np.linalg.norm(traj_fix_time[10] - traj_fix_time[5]) * 2.0 # average speed between 0.5s-1s
        if desired_speed < self.brake_threshold: # brake if desired speed is small
            throttle = 0.0
            brake = 1.0
            self.speed_window = self.speed_window[-self.speed_n:] + [0.0]
        elif desired_speed < current_speed: # brake if desired speed is smaller than current speed
            throttle = 0.0
            brake = self.brake_b + self.brake_k * (current_speed - desired_speed)
            brake = np.clip(brake, 0.0, 1.0)
            self.speed_window = self.speed_window[-self.speed_n:] + [0.0]
        else: # PID control for throttle
            speed_error = np.clip(desired_speed - current_speed, 0.0, 1.0)
            self.speed_window = self.speed_window[-self.speed_n:] + [speed_error]
            speed_integral = np.mean(self.speed_window)
            speed_derivative = self.speed_window[-1] - self.speed_window[-2] if len(self.speed_window) > 1 else 0
            throttle = self.speed_k_p * speed_error + self.speed_k_d * speed_derivative + self.speed_k_i * speed_integral
            throttle = np.clip(throttle, 0.0, 1.0)
            brake = 0.0
        # get reference point for steering condictioned on current speed
        point_idx_for_heading = int(np.clip(current_speed * 0.15, 1, len(traj_fix_dist)-1))
        heading_error = np.arctan2(traj_fix_dist[point_idx_for_heading, 1], traj_fix_dist[point_idx_for_heading, 0]) 
        # normalize heading
        while heading_error < -np.pi:
            heading_error += 2 * np.pi
        while heading_error > np.pi:
            heading_error -= 2 * np.pi  
        # PID control for steering
        self.steer_window = self.steer_window[-self.steer_n:] + [heading_error]
        steer_integral = np.mean(self.steer_window)
        steer_derivative = self.steer_window[-1] - self.steer_window[-2] if len(self.steer_window) > 1 else 0
        steer = self.steer_k_p * heading_error + self.steer_k_d * steer_derivative + self.steer_k_i * steer_integral
        steer = np.clip(steer, -1.0, 1.0) 
  
        return steer, throttle, brake
              

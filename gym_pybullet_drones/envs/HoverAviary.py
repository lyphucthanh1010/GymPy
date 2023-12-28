import numpy as np

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
from gym import spaces

class HoverAviary(BaseRLAviary):
    """Single agent RL problem: hover at position."""

    ################################################################################
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM
                 ):
        """Initialization of a single agent RL environment.
	
        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=255, shape=(3, 3), dtype=np.uint8)
        self.object_position = [0, 0, 0]
        self.TARGET_POS = np.array([0,0,1])
        self.EPISODE_LEN_SEC = 8
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )

    ################################################################################
    
    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        state = self._getDroneStateVector(0)
        ret = max(0, 2 - np.linalg.norm(self.TARGET_POS-state[0:3])**4)
        return ret

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)
        if np.linalg.norm(self.TARGET_POS-state[0:3]) < .0001:
            return True
        else:
            return False
        
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        state = self._getDroneStateVector(0)
        if (abs(state[0]) > 1.5 or abs(state[1]) > 1.5 or state[2] > 2.0 # Truncate when the drone is too far away
             or abs(state[7]) > .4 or abs(state[8]) > .4 # Truncate when the drone is too tilted
        ):
            return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False
	
    ################################################################################
    def _calculate_next_observation(current_observation, action, target_position):
        speed_factor = 0.1
        dx,dy,dz = action
        direction_to_target = target_position - np.array(current_observation)
        distance_to_target = np.linal.norm(direction_to_target)
        if distance_to_target > 0.1:
            direction_to_target /= distance_to_target
            movement = speed_factor * direction_to_target
            
            next_observation = current_observation + movement
        else:
            next_observation = current_observation
       
        return next_observation.tolist()
        
    def _calculate_reward(current_observation, action):
    # Logic để tính toán reward dựa trên observation hiện tại và hành động
    # Ví dụ đơn giản, sử dụng giá trị tuyệt đối của action làm reward
        reward = abs(action)
        return reward
    
    def _check_if_done(current_observation):
    # Logic để kiểm tra xem môi trường đã kết thúc hay chưa
    # Ví dụ đơn giản, kiểm tra xem observation có vượt quá một ngưỡng nào đó hay không
        done = current_observation > 10  # Ví dụ ngưỡng là 10
        return done
    
    def _additional_information(current_observation, action):
    # Logic để trả về thông tin bổ sung
    # Ở đây, chúng ta chỉ đơn giản trả về một tuple gồm observation và action
        info = {'current_observation': current_observation, 'action': action}
        return info
    def _set_object_position(self, object_position):
        # Thiết lập vị trí của vật thể trong môi trường
        self.object_position = object_position
        print(f"Object position set to: {object_position}")
        
    def _get_object_position(self):
        # Lấy vị trí của vật thể từ môi trường
        return self.object_position
    
    def _get_drone_observation(self):
        # Giả sử môi trường cung cấp thông tin vị trí và tốc độ của drone
        drone_position = [1.0, 2.0, 3.0]  # Giả sử vị trí của drone là [1.0, 2.0, 3.0]
        drone_velocity = [0.5, -0.2, 1.0]  # Giả sử tốc độ của drone là [0.5, -0.2, 1.0]

        # Tạo observation của drone từ vị trí và tốc độ
        drone_observation = drone_position + drone_velocity
        return drone_observation
        
    ##################################################################################
    def _reset(self, seed = 42):
    # Lấy thông tin về vị trí vật thể
        object_position = np.array(self.get_object_position())
    
    # Lấy observation ban đầu của drone
        drone_observation = np.array(self.get_drone_observation())
    # Nếu kích thước của drone_observation hoặc object_position không đúng, điều chỉnh chúng thành kích thước mong muốn
        if drone_observation.size != 9 or object_position.size != 18:
            drone_observation = np.zeros(9)  # Điều chỉnh kích thước của drone_observation
            object_position = np.zeros(18)   # Điều chỉnh kích thước của object_position

    # Nối các thông tin lại với nhau
        drone_observation_with_object = np.concatenate([drone_observation, object_position])
    
    # Đảm bảo kích thước cuối cùng của drone_observation_with_object là (1, 27)
        drone_observation_with_object = drone_observation_with_object.reshape((1, 27))
        initial_reward = 0
        done = False
        info = {} 
        return drone_observation_with_object, (initial_reward, done, info)


    #################################################################################
    
    
    def _step(self, action):
    # Use the current observation and action to calculate the next state
        next_observation = self._calculate_next_observation(self.current_observation, action)
    
    # Use the next state and action to calculate the reward
        reward = self._calculate_reward(self.current_observation, action)
    
    # Check if the episode is done based on the next state
        done = self._check_if_done(next_observation)
    
    # Additional information if needed
        info = self._additional_information(self.current_observation, action)
    
    # Update the current observation to the next observation for the next step
        self.current_observation = next_observation
    
        return next_observation, reward, done, info
      ##############################################################################
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years
    

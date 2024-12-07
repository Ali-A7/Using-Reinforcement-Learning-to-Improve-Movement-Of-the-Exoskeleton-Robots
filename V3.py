#import gym
import gymnasium as gym # Import gymnasium instead of gym
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

class ExoskeletonEnv(gym.Env):
    eI = 0

    def __init__(self):
        super(ExoskeletonEnv, self).__init__()

        # Action space: 2 motor torques (for shoulder and elbow)
        self.action_space = gym.spaces.MultiDiscrete([3,3])

        # Observation space: human torques (2), angular velocities (2), previous human torques (2)
        self.observation_space = gym.spaces.Box(low=-58, high=74, shape=(2,), dtype=np.float32)

        # Timer
        self.times = 0
        self.moon = 0
        self.sed = 0
        #self.resetcount = 0
        self.sum_reward_3 = 0
        self.sum_reward_4 = 0

        # Load dataset for human torque and scaling
        #self.df = pd.read_csv(r'C:\Users\Ali\Desktop\data_exo\DeepL_data\Deep\Deep_Txh_Av.csv')
        self.df = pd.read_csv(r'C:\Users\Ali\Desktop\data_to_run\Exo_full_data.csv')

        # Extract input features and output labels for scaling
        X = self.df[['Tx3', 'Tx4', 'Th3', 'Th4']].values
        y = self.df[['Av3', 'Av4']].values

        # Normalize the features and labels
        self.scaler_X = MinMaxScaler(feature_range=(-1, 1))
        self.scaler_y = MinMaxScaler(feature_range=(-1, 1))

        X = self.scaler_X.fit_transform(X)
        y = self.scaler_y.fit_transform(y)

        # Load the pre-trained LSTM model
        time_steps = 100
        self.model = Sequential()
        self.model.add(LSTM(50, return_sequences=True, input_shape=(time_steps, X.shape[1])))
        self.model.add(LSTM(50))
        self.model.add(Dense(2))
        self.model.compile(optimizer='adam', loss='mse')
        #self.model.load_weights(r'C:\Users\Ali\Desktop\data_exo\DeepL_data\Deep\model_weights.weights.h5')
        self.model.load_weights(r'C:\Users\Ali\Downloads\lstms_model.weights.h5')


        # Initialize state and previous inputs
        #self.state = np.zeros(4)
        #self.previous_inputs = []
                # Reset the state of the environment to an initial state
        self.state = np.zeros(2)
        self.previous_inputs = []
        self.angular_velocity_shoulder = 0
        self.angular_velocity_elbow = 0
        self.times = 0
        self.previous3 = 0
        self.previous4 = 0
        self
            # Initialize the first 10 inputs for the LSTM model
        for i in range(100):
            # Get initial motor torques and human torques from the dataset
            motor_torque_shoulder, motor_torque_elbow = self.df.at[i, 'Tx3'], self.df.at[i, 'Tx4']
            human_torque_shoulder, human_torque_elbow = self.df.at[i, 'Th3'], self.df.at[i, 'Th4']

            # Prepare input for the model
            initial_input = np.array([motor_torque_shoulder, motor_torque_elbow, human_torque_shoulder, human_torque_elbow])
            self.previous_inputs.append(initial_input)

        # Normalize the inputs
        self.previous_inputs = np.array(self.previous_inputs)
        self.previous_inputs = self.scaler_X.transform(self.previous_inputs)
    def reset(self,seed = None ):

        # Reset the state of the environment to an initial state
        #self.state = np.zeros(4)
        #self.previous_inputs = []

        self.moon = 0
        #self.resetcount += 1
        #print('#############',self.resetcount,'##############')

        #import pdb
        #pdb.set_trace()

        return self.state,{}

    def step(self, action):

        # print(self.times)
        print('**********************',self.times,self.sed,'*************************')
        print('#############',self.moon,'##############')



        # Extract components from state
        #human_torque_shoulder_prev, human_torque_elbow_prev = self.df.at[self.times - 1, 'Th3'], self.df.at[self.times - 1, 'Th4']
        human_torque_shoulder, human_torque_elbow = self.df.at[self.times, 'Th3'], self.df.at[self.times, 'Th4']
        human_torque_shoulder2, human_torque_elbow2 = self.df.at[self.times + 1, 'Th3'], self.df.at[self.times + 1, 'Th4']
        #angular_velocity_shoulder, angular_velocity_elbow = self.state[2:4]

        # Apply action (motor torques)
        motor_torque_shoulder, motor_torque_elbow = action
        motor_torque_shoulder -=1
        motor_torque_shoulder = motor_torque_shoulder*0.1
        motor_torque_shoulder = motor_torque_shoulder + self.state[0]
        motor_torque_elbow -=1
        motor_torque_elbow = motor_torque_elbow*0.1
        motor_torque_elbow = motor_torque_elbow + self.state[1]

        # Prepare new input for the model
        new_input = np.array([motor_torque_shoulder, motor_torque_elbow, human_torque_shoulder, human_torque_elbow])
        #self.previous_inputs.append(new_input)
        self.previous_inputs = np.append(self.previous_inputs[1:], [new_input], axis=0)

        #if len(self.previous_inputs) > 10:
         #   self.previous_inputs.pop(0)

        # Convert to numpy array and normalize
        # Handle the case where there are not enough previous inputs
        #if len(self.previous_inputs) < 10:
            # Pad the input sequence with zeros to reach the required length
            #padding = np.zeros((10 - len(self.previous_inputs), 4))
            #input_sequence = np.concatenate((padding, np.array(self.previous_inputs)))
        #else:

        #input_sequence = np.array(self.previous_inputs)
        #input_sequence = self.scaler_X.transform(input_sequence.reshape(-1, 4)).reshape(1, 10, 4)

        # Convert to numpy array and normalize
        input_sequence = self.previous_inputs.reshape(1, 100, 4)

        # Predict the angular velocities using the LSTM model
        predicted_angular_velocities = self.model.predict(input_sequence)
        predicted_angular_velocities = self.scaler_y.inverse_transform(predicted_angular_velocities)[0]

                # Update angular velocities in the state
        angular_velocity_shoulder = predicted_angular_velocities[0]
        angular_velocity_elbow = predicted_angular_velocities[1]

        # Calculate jerk for both shoulder and elbow
        jerk_shoulder = self.calculate_jerk(angular_velocity_shoulder, self.previous3)
        jerk_elbow = self.calculate_jerk(angular_velocity_elbow, self.previous4)



        # Measure w based on motor torque and PID-ES output
        Ta_shoulder = self.PID_ES(human_torque_shoulder)
        Ta_elbow = self.PID_ES(human_torque_elbow)

        #new_measured_w_shoulder = self.measure_w(motor_torque_shoulder, Ta_shoulder)
        #new_measured_w_elbow = self.measure_w(motor_torque_elbow, Ta_elbow)

        # Update the state
        Ta_shoulder2 = self.PID_ES(human_torque_shoulder2)
        Ta_elbow2 = self.PID_ES(human_torque_elbow2)
        self.state = np.array([
            Ta_shoulder2, Ta_elbow2
        ])

        # Reward function: penalize high jerk and high w difference
        result = (jerk_shoulder + jerk_elbow )



        reward = self.calculate_reward(result)
        self.sum_reward_3 += jerk_elbow
        self.sum_reward_4 += jerk_shoulder

        #print('++++++++++++++++++++++++++','new_measured_w_shoulder:',new_measured_w_shoulder ,'new_measured_w_elbow:', new_measured_w_elbow ,'jerk_shoulder:', jerk_shoulder ,'jerk_elbow:',jerk_elbow ,'result:',result , '++++++++++++++++++++++++++++')
        print('+++','Ta_elbow:',Ta_elbow ,'motor_torque_elbow:', motor_torque_elbow ,'motor_torque_shoulder:', motor_torque_shoulder ,'Ta_shoulder:',Ta_shoulder ,'result:',result ,'sum_rewardelbow',self.sum_reward_3 , 'sum_reward_shoulder',self.sum_reward_4)
        print('+++','angular_velocity_shoulder:',angular_velocity_shoulder ,'self.previous3:', self.previous3 ,'angular_velocity_elbow:', angular_velocity_elbow ,'self.previous4:',self.previous4 ,'reward:',reward , '+++')
        
        # Update angular velocities in the state
        self.previous3 = predicted_angular_velocities[0]
        self.previous4 = predicted_angular_velocities[1]
        #self.times += 1
        # Increment the timer
        if self.times <= 100000:
          self.times += 1
        else:
          self.times = 0
          self.sed += 1
        
        self.moon += 1
        # Define a condition for termination if needed
        # Force done after each step
        #done = False  # This triggers an environment reset after each step
        #truncated = False  # Since we are forcing `done`, truncated will also be True.
        # Define a condition for termination if needed
        done = self.moon >= 300
        truncated = done  # Set truncated to True if done is True (time limit reached)
        terminated = False # Set terminated to False as there is no true ending condition

        return self.state, reward, terminated, truncated, {}

    def calculate_jerk(self, angular_velocity, new_angular_velocity):
        jerk = (new_angular_velocity - angular_velocity) ** 2
        return jerk


    def calculate_reward(self,result):
        target_value = 0
        distance_from_target = abs(result - target_value)

        # Positive reward if within a small range of zero
       # if distance_from_target == 0:
        #    penalty = 10
        #else :
         #   penalty = -np.log(distance_from_target)

  
        # Positive reward if within a small range of zero
        if distance_from_target == 0:
            penalty = 100
        elif distance_from_target > 0 and distance_from_target < 1:
            penalty = 1/(distance_from_target)
        elif distance_from_target == 1:
            penalty = 1    
        elif distance_from_target < 3 and distance_from_target > 1:
            penalty = 1/(distance_from_target)   
        else :
            penalty = -distance_from_target

        return penalty

    def measure_w(self, exo_torque, Ta):
        # Measure the difference between human torque and PID-ES torque
        w = exo_torque - Ta
        return w

    def PID_ES(self, human_torque):
        if abs(human_torque) > 1:
            if human_torque > 0:
                human_torque -= 1
            else:
                human_torque += 1
        else:
            human_torque = 0

        if abs(self.eI) < 1:
            self.eI += human_torque * 0.01
        else:
            if self.eI > 0:
                self.eI += (human_torque-1) * 0.01
            else:
                self.eI += (human_torque+1) * 0.01

        Ta = (2 * human_torque) + (5 * self.eI)
        return Ta
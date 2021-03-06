from unityagents import UnityEnvironment

class Env():
    """Simplifies the interaction with the environment."""
    
    def __init__(self, file_name="Banana_Windows_x86_64/Banana.app"):
        """Iniliaze new unity enviroment
        
        Params
        ======
            file_name (str): the location of the enviroment to load
            """
        self.env = UnityEnvironment(file_name=file_name)

        # get the default brain
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        env_info = self.env.reset(train_mode=False)[self.brain_name]
        
        #get the action space
        self.action_size = self.brain.vector_action_space_size
        
        #get the state space
        state = env_info.vector_observations[0]
        self.state_size = len(state)
        
        print("state_size:", self.state_size, " action_size:", self.action_size )
        
    def reset(self, train_mode=False):
        """Reset the unity environment and returns the current state
        
        Params
        ======
            train_mode (bool): Whether you want to set training mode or not
            """
        # reset the environment
        env_info = self.env.reset(train_mode=train_mode)[self.brain_name] 
        states = env_info.vector_observations
        return states
    
    def execute(self, actions):
        """Execute a step on the environment corresponding to the action received.
        Returns the next state, the reward received and a boolean value "done" that 
        indicates if the environment has come to a terminal state.
        
        Params
        ======
            action (int): The index for the action to perform.
            """
        #action = int(action)
        env_info = self.env.step(actions)[self.brain_name]        # send the action to the environment
        next_states = env_info.vector_observations             # get the next state
        rewards = env_info.rewards                             # get the reward
        dones = env_info.local_done                            # get the flag of done
        return next_states, rewards, dones
    
    def close(self):
        """ Closes the environment        """
        self.env.close()
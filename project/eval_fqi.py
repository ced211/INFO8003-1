from sklearn.externals.joblib import dump, load
import numpy as np
import matplotlib
from catcher import ContinuousCatcher
import matplotlib.pyplot as plt

def evaluate(model_path,EPISODE_NUMBER = 20, TIME_LIMITE = 1000, env = ContinuousCatcher()):
    """
    Evaluate the model: return the mean of cumulative reward 
    """
    actions = [-80,-60,-40,-20,-10,-5,0,5,10,20,40,60,80]
    model = load(model_path)
    reward_sum = 0
    episode_number = 0
    previous_observation = env.reset()
    sum_reward_list = []        
    time = 0        
    while episode_number < EPISODE_NUMBER:
        # give the mean and action given an obervation.
        action = 0
        action_value = 0
        for a in actions:
            value = model.predict(np.append(previous_observation,a).reshape(1,-1))
            if  value > action_value:
                action = a
                action_value = value
        # give the observation and reward given an action.
        observation, reward, done = env.step([action])
        previous_observation = observation
        reward_sum += reward
        time += 1
        if done or time > TIME_LIMITE:
            print("episode" + str(episode_number))
            print("reward" + str(reward_sum))
            sum_reward_list.append(reward_sum)
            print(sum_reward_list)
            reward_sum = 0
            episode_number += 1
            previous_observation = env.reset()	
            time = 0
            env.reset()
            
    sum_reward_array = np.asarray(sum_reward_list)
    return np.mean(sum_reward_array), np.std(sum_reward_array)
	
def plot_fqi(path_std, path_mean):
    std_list = np.load(path_std)
    mean_list = np.load(path_mean)
    x = range(1, len(mean_list)+1)
    plt.errorbar(x, mean_list, std_list, linestyle='None', marker='^')
    plt.show()
if __name__ == "__main__":
    mean_list, std_list = [], []
    for i in range(21):
        model_path = "./model/Q"+ str(i+2) + ".sav"
        mean,std = evaluate(model_path)
        mean_list.append(mean)
        std_list.append(std)
    np.save("fqi_std_list",np.asarray(std_list))
    np.save("fqi_mean_list",np.asarray(mean_list))
    plot_fqi("fqi_std_list.npy", "fqi_mean_list.npy")


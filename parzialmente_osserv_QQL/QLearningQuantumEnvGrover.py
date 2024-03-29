from environments import QuantumToyEnv
from algorithms import QuantumQLearning, runEnvironment
import time
import numpy as np
import matplotlib.pyplot as plt



# Instantiate environment
env= QuantumToyEnv()
#display(env.qc.draw('mpl'))



# Q-Learning Algorithm hyperparameters
gamma= 0.99 # Discount factor
MaxSteps= 200 # Maximum number of iterations
eps0= 0.5 # Initial epsilon value for e-greedy policy
epsf= 0.001 # Final epsilon value for e-greedy policy
epsSteps= MaxSteps # Number of steps to decrease e-greedy epsilon from eps0 to epsf
alpha= 0.2 # Learning rate
exploration0= 1.0
explorationf= 0.01
explorationSteps= MaxSteps

show= True # To show current step during simulation
import pickle
# Execute Q-Learning algorithm
t0= time.time()
policy, Niter, rewards = QuantumQLearning(env, MaxSteps, eps0, epsf, epsSteps, alpha, gamma, exploration0, explorationf, explorationSteps, show)
with open('qlearning_model0202.pkl', 'wb') as f:
    pickle.dump(policy, f)
tf= time.time()

print ("POLITICHE")
plt.plot(rewards)
plt.xlabel('Step')
plt.ylabel('Reward')
plt.show()


# Show results
print('Q-Learning stopped after {} iterations'.format(Niter))
print('The execution time was: {} s.'.format(tf-t0))
print('The policy obtained is:')
for s in range(len(policy)):
    print('\tFor state {}: Execute action {}'.format(s, policy[s]))
    
    

# Execute policy
MaxIterations= 200 # Number of iterations for test
MaxTests= 100
RewardSet= []
weights= []
for test in range(MaxTests):
    print('Test environment #{}/{}'.format(test+1, MaxTests))
    R, t_total= runEnvironment(env, policy,weights, MaxIterations, showIter=False)
    
    RewardSet.append(R)
    


print('Total Reward over {} experiences: {}'.format(MaxIterations, np.mean(RewardSet)))

plt.hist(RewardSet)
plt.xlabel('Total reward')
plt.ylabel('# Times obtained')
plt.title('Q-Learning (Quantum env.)')
plt.show()

with open('qlearning_model0202.pkl' , 'rb') as f:
    policy = pickle.load(f)
    





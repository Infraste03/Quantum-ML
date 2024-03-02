from envMulti import  MultiAgentQuantumToyEnv
from algoMulti import  runEnvironment, MultiAgentQLearning
import time
import numpy as np
import matplotlib.pyplot as plt
num_agents= 4

# Instantiate environment
#env= QuantumToyEnv()
env=MultiAgentQuantumToyEnv(num_agents)
#display(env.qc.draw('mpl'))


# Q-Learning Algorithm hyperparameters
gamma= 0.99 # Discount factor
MaxSteps= 200 # Maximum number of iterations
eps0= 0.5 # Initial epsilon value for e-greedy policy
epsf= 0.001 # Final epsilon value for e-greedy policy
epsSteps= MaxSteps # Number of steps to decrease e-greedy epsilon from eps0 to epsf
alpha= 0.2 # Learning rate



show= True # To show current step during simulation

# Execute Q-Learning algorithm
t0= time.time()
#policy, Niter= QLearning(env, MaxSteps, eps0, epsf, epsSteps, alpha, gamma, show)
policies = MultiAgentQLearning(env, num_agents, MaxSteps, eps0, epsf, epsSteps, alpha, gamma, show)
tf= time.time()

print ("politiche multiagente", policies)



# Show results
#print('Q-Learning stopped after {} iterations'.format(Niter))
print('The execution time was: {} s.'.format(tf-t0))
print('The policy obtained is:')
for s in range(len(policies)):
    print('\tFor state {}: Execute action {}'.format(s, policies[s]))
    
    

# Execute policy
MaxIterations= 200 # Number of iterations for test
MaxTests= 100
RewardSet= []
for test in range(MaxTests):
    print('Test environment #{}/{}'.format(test+1, MaxTests))
    R, t_total= runEnvironment(env, policies, MaxIterations, showIter=False)
    RewardSet.append(R)
    


print('Total Reward over {} experiences: {}'.format(MaxIterations, np.mean(RewardSet)))

plt.hist(RewardSet)
plt.xlabel('Total reward')
plt.ylabel('# Times obtained')
plt.title('Q-Learning (Quantum env.)')
plt.show()
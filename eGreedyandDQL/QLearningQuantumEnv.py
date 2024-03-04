import time
import numpy as np
import matplotlib.pyplot as plt

#* in questo file posso importare algo2 nel caso in cui voglia testare il codice della funzione DQLearning 
#* e env2 nel caso in cui voglia testare il codice con le modifiche per Us, Ua, Ut e Ur
from env2 import QuantumToyEnv
from algo2 import DQLearning, runEnvironment
# Instantiate environment
env= QuantumToyEnv()
#display(env.qc.draw('mpl'))



# Q-Learning Algorithm hyperparameters
gamma= 0.99 # Discount factor
MaxSteps= 500 # Maximum number of iterations #^ prima era 200
eps0= 0.9 # Initial epsilon value for e-greedy policy #^ prima era 0.5
epsf= 0.001 # Final epsilon value for e-greedy policy
epsSteps= MaxSteps # Number of steps to decrease e-greedy epsilon from eps0 to epsf
alpha= 0.5 # Learning rate #^ prima era 0.2

show= True # To show current step during simulation

# Execute Q-Learning algorithm
t0= time.time()
state_size = env.nS
print("env" , env.nS)
action_size = env.nqA

#exit(0)
policy, Niter= DQLearning(env, state_size, action_size, MaxSteps, eps0, epsf, epsSteps, alpha, gamma, show)
tf= time.time()


states = np.arange(state_size)[:, np.newaxis]
predictions = policy.predict(states)
print("predictions ", predictions)
# Show results
print('Q-Learning stopped after {} iterations'.format(Niter))
print('The execution time was: {} s.'.format(tf-t0))
print('The policy obtained is:')
print("count stati ", states.shape)


for s, pred in zip(states, predictions):
    print('\tFor state {}: Execute action {}'.format(s, np.argmax(pred)))
    
    

# Execute policy
MaxIterations= 200 # Number of iterations for test
MaxTests= 500
RewardSet= []
for test in range(MaxTests):
    print('Test environment #{}/{}'.format(test+1, MaxTests))
    R, t_total= runEnvironment(env, policy, MaxIterations, showIter=False)
    RewardSet.append(R)
    


print('Total Reward over {} experiences: {}'.format(MaxIterations, np.mean(RewardSet)))

plt.hist(RewardSet)
plt.xlabel('Total reward')
plt.ylabel('# Times obtained')
plt.title('Q-Learning (Quantum env.)')

plt.show()

import numpy as np
from collections import defaultdict
import time
from qiskit import QuantumRegister, QuantumCircuit



def runEnvironment(env, policies, iterations, showIter=False):
    num_agents = 4
    s = env.reset()
    if isinstance(s, int):
        s = [s]
    s = s[:num_agents]
    policies = policies[:num_agents]
    if len(s) != len(policies):
        if len(s) > len(policies):
            # Extend policies with the last policy
            policies += [policies[-1]] * (len(s) - len(policies))
        else:
            # Truncate policies
            policies = policies[:len(s)]
    assert len(s) == len(policies), "Number of states must be equal to number of policies"
    

    R = [0 for _ in policies]  # R is now a list of rewards for each agent
    t0 = time.time()
    for it in range(iterations):
        a = []
        for state, policy in zip(s, policies):
            try:
                a.append(int(policy[state]))
            except IndexError:
                print(f"Invalid state {state} for policy with {len(policy)} states")
                a.append(0)  # or some other default action
        
        # If a has more than 4 actions, truncate it
        if len(a) > 4:
            a = a[:4]

        # If a has less than 4 actions, extend it with the last action
        elif len(a) < 4:
            a = a + [a[-1]] * (4 - len(a))

        sp, r = env.step(a)
        
        R = [reward + r for reward, r in zip(R, r)]
        s = sp
        
        if showIter:
            print('Test at iteration {}/{} with R={}'.format(it+1, iterations, R))

    tf = time.time()
    t_total = tf - t0
    return R, t_total

def ValueIteration(env, gamma, iterations=20, convThres=1e-20):
    
    # Table containing V(s) values initialized to 0
    Vtable= np.zeros(env.numberOfStates())
    
    end= False # Loop Stopping criterion 
    converge= False # To know if the algorithm has converged
    
    
    it= 0 # Current iteration
    
    # Value Iteration main loop
    while not end: 
        auxVtable= Vtable.copy()
        
        # Update table Q(s,a)= sum_{s'} p(s'|s,a)*(r(s,a,s')+gamma*V(s))
        for s in range(env.numberOfStates()):
            Qtable= np.zeros(env.numberOfActions())
            
            for a in range(env.numberOfActions()):
                for sp in range(env.numberOfStates()):
                    
                    # Get transition probability from (s,a)->sp
                    p= env.transitionProb(s, a, sp)
                    if p>0.0: # Update Q-Table
                        Qtable[a]+= p*(env.rewardValue(s,a,sp) + gamma*auxVtable[sp] )
            
            # Update V(s)= max_{a} Q(s,a)
            Vtable[s]= np.max(Qtable)
        
        # Prepare next iteration
        it+= 1
        
        # Check convergence
        converge= np.max(np.fabs(Vtable-auxVtable)) <= convThres
        
        # Stopping criterion
        end= (it>=iterations or converge)
        
    #Return the V-table, the number of iterations performed and if the algorithm converged
    return Vtable, it, converge


def ExtractPolicyFromVTable(env, Vtable, gamma):
    
    # Create policy table
    policy= np.zeros(len(Vtable), dtype=int) 
    
    for s in range(len(Vtable)):
        
        # Create Q-table
        Qtable= np.zeros(env.numberOfActions())
        for a in range(env.numberOfActions()):
            for sp in range(env.numberOfStates()):
                
                # Get transition probability from (s,a)->sp
                p= env.transitionProb(s, a, sp)
                if p>0.0:
                    Qtable[a]+= p*(env.rewardValue(s,a,sp) + gamma*Vtable[sp] )
        # Update policy
        policy[s]= np.argmax(Qtable)
    return policy

    
def eGreedyPolicy(env, S, AgentPolicy, Q, epsilon):
    
    if np.random.rand() < epsilon: # Random uniform policy
        return np.random.randint(low= 0, high=env.numberOfActions()) 
    
    else:
        return AgentPolicy(env, S, Q)
    

# Agent Policy that selects the action with maximum Q-value
# param env: The environment
# param S: The current state observation
# param Q: the Q-table Q[(s,a)]-> Value of state-action pair (s,a)
#
# Returns an action to be executed in the environment
def AgentPolicy(env, S, Q):
    return np.argmax([Q[(S, a)] for a in range(env.numberOfActions())]) 


def QLearning(env, MaxSteps, eps0, epsf, epsSteps, alpha, gamma, show=False):
    
    # Initialize epsilon
    epsilon= eps0
    
    # Initialize Q-table  Q(s,a)= 0 
    Qini= defaultdict(float)
    
    end= False # Stopping criterion
    it= 0 # Number of steps performed
    
    # Initialize environment and get initial state
    s= env.reset()
   # s = s % len(policy)
    
    # Q-Learning cycle
    while not end:

        if show:
            print('Running step {}'.format(it+1))

        Q= Qini.copy()
        
        # Select action for current state
        
        a1 = eGreedyPolicy(env, s, AgentPolicy, Q, epsilon)
        a2 = eGreedyPolicy(env, s, AgentPolicy, Q, epsilon)
        a3 = eGreedyPolicy(env, s, AgentPolicy, Q, epsilon)
        a4 = eGreedyPolicy(env, s, AgentPolicy, Q, epsilon)
        
        # Execute action in environment
        sp, r= env.step([a1, a2,a3, a4])
        # In your QLearning function in algorithms.py
        s, r = env.step([a1, a2,a3, a4])
        #sp, r= env.step([a1, a2])
        # r = r.item()  # Remove this line
        
        # Get best known action ap
        #QValues= [Q[(sp, a_)] for a_ in range(env.numberOfActions())]
        QValues= [Q[(tuple(sp), a_)] for a_ in range(env.numberOfActions())]
        ap= np.argmax( QValues ) 
        
        # Q-Learning update rule
        # Ensure s and sp are tuples before using them as keys
        s = tuple(s)
        sp = tuple(sp)

        # Ensure r is a float before using it in the expression
        
        r = r[0]
        #print ("ciaoaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", r)

        # Now you can use r in your expression
        Q[(s, a1)] += alpha * (r + gamma * Q[(sp, ap)] - Q[(s, a1)])
        Q[(s, a2)] += alpha * (r + gamma * Q[(sp, ap)] - Q[(s, a2)])
        Q[(s, a3)] += alpha * (r + gamma * Q[(sp, ap)] - Q[(s, a3)])
        Q[(s, a4)] += alpha * (r + gamma * Q[(sp, ap)] - Q[(s, a4)])


        # Prepare next step
        s= sp


        # Reset env if required
        if env.StoppingCriterionSatisfied():
            sp= env.reset()
        
        # Update epsilon
        epsilon= max(epsf, eps0+it*(epsf-eps0)/epsSteps)
        
        # prepare next iteration
        Qini= Q
        it+= 1
        if it>=MaxSteps:
            end= True
            
            
    # calculate policy
    policy= np.zeros(env.numberOfStates(), dtype=int) 
    for s in range(env.numberOfStates()):
        action= int(AgentPolicy(env, s, Qini))
        policy[s]= action
        
    
    return policy, it


def MultiAgentQLearning(env, num_agents, MaxSteps, eps0, epsf, epsSteps, alpha, gamma, show=False):
    policies = []
    for _ in range(num_agents):
        policy, _ = QLearning(env, MaxSteps, eps0, epsf, epsSteps, alpha, gamma, show)
        policies.append(policy)
    return policies

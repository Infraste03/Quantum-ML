import numpy as np
from collections import defaultdict
import time


#^ INTRODUCO UN FATTORE DI ESPLORAZIONE VARIABILE  PER IL Q-LEARNING
#* invece di utilizzare un fattore epsilon fisso per la politica e-greedy, introduco un fattore che
#*diminuisce nel tempo. Questo lo faccio perchè permetterebbe all agente di esplorare l' ambiente 
#*inizialemnte e, mano man che acquista esperienza, di sfruttare le informazioni acquisite per massimizzare la rimcpensa  


def runEnvironment(env, policy, weights, MaxIterations, showIter=False):
    # function implementation
    
    s= env.reset()
    R= 0
    t0= time.time()
    for it in range(MaxIterations):
        
        a= int( policy[s] )
        sp, r= env.step(a)
        
        R+= r.squeeze()
        s= sp
        
        if showIter:
            print('Test at iteration {}/{} with R={}'.format(it+1, MaxIterations, R))


    tf= time.time()
    t_total= tf-t0
    return R, t_total

#       converge: Boolean to know if the algorithm has converged
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



#? IN QUESTO CODICE SI INTRODUCE UN FATTORE DI ESPLORAZIONE VARIABILE CHE DIMINUISCE NEL TEMPO 
#? PER FARE QUESTO INTRODUCENDO UN NUOVO PARAMENTRO CHE VIENE AGGIORNATO AD ONGI INTERAZIONE 

#*In questo codice, viene eseguito il circuito di Grover 
#*e il risultato viene utilizzato per determinare se l'agente dovrebbe eseguire un'azione di esplorazione o di sfruttamento.

from qiskit import QuantumCircuit, execute, Aer

def QuantumQLearning(env, MaxSteps, eps0, epsf, epsSteps, alpha, gamma,exploration0, explorationf,explorationSteps,  show=False):
    
    # Initialize epsilon
    epsilon= eps0
    #Inizializzazione del fattore di esplorazione 
    exploration = exploration0
    # Initialize Q-table  Q(s,a)= 0 
    Qini= defaultdict(float)
    end= False # Stopping criterion
    it= 0 # Number of steps performed
    rewards = []  # Initialize rewards list
    # Initialize environment and get initial state
    s= env.reset()
    
    # Q-Learning cycle
    while not end:

        if show:
            print('Running step  {}'.format(it+1))
        Q= Qini.copy()
        
        # Select action for current state
        # Create a quantum circuit for Grover's algorithm
        grover_circuit = QuantumCircuit(2, 2)
        grover_circuit.h([0,1])
        grover_circuit.cz(0,1)
        grover_circuit.h([0,1])
        grover_circuit.z([0,1])
        grover_circuit.cz(0,1)
        grover_circuit.h([0,1])
        grover_circuit.measure([0,1], [0,1])

        # Execute the quantum circuit to select the action
        result = execute(grover_circuit, backend=Aer.get_backend('qasm_simulator')).result()
        counts = result.get_counts(grover_circuit)

        # Use the result of the Grover's algorithm to influence the action selection
        grover_result = max(counts, key=counts.get)

        # Select action for current state
        if grover_result == '00' or np.random.rand() < epsilon:  # Exploration: choose a random action
            a = np.random.choice(env.numberOfActions())
        else:  # Exploitation: choose the action with the highest Q-value
            QValues= [Q[(s, a_)] for a_ in range(env.numberOfActions())]
            a = np.argmax(QValues)

        # Execute action in environment
        sp,r = env.step(a) 
        r  = r.squeeze() 
        rewards.append(r)  # Add reward to total reward
        
        # Get best known action ap
        #QValues= [Q[(sp, a_)] for a_ in range(env.numberOfActions())]
        ap= np.argmax( QValues ) 
        
        # Q-Learning update rule
        Q[(s,a)]+= alpha*(r+gamma*Q[(sp, ap)] - Q[(s,a)])
        # Prepare next step
        s= sp

        # Reset env if required
        if env.StoppingCriterionSatisfied():
            sp= env.reset()
        
        # Update epsilon
        epsilon= max(epsf, eps0+it*(epsf-eps0)/epsSteps)
        
        #* aggioranmento del fattore di esplorazione
        exploration = max(explorationf, exploration0+it*(explorationf-exploration0)/explorationSteps)
        
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
    
    return policy, it, rewards

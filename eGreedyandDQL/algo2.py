import numpy as np
from collections import defaultdict
import time
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


#*MODIFICA CODICE PER IL PROGETTO:
 #*in questo caso è stato modificata la funzione DQLearning, seguendo i seguenti criteri:
 #* 1) Modularità: La funzione create_model è separata dalla funzione DQLearning,
 #*2) Uso di Deep Learning: permette di gestire spazi di stato e azione molto grandi che non potrebbero essere gestiti con metodi tabulari.
 #*3) Politica Epsilon-Greedy & Decadimento di Epsilon
 #*4) Aggiornamento online: aggiornando il tuo modello ad ogni passo, il che può aiutare l'agente ad adattarsi rapidamente a nuove informazion
 #*5) Uso di un criterio di arresto: Il  codice controlla se un criterio di arresto è soddisfatto e resetta l'ambiente 
 #*se necessario. Questo è utile per problemi in cui l'agente deve ripetere un compito molte volte
 #*6) Controllo del numero massimo di passaggi: Il  codice termina dopo un numero massimo di passaggi, il che è utile per evitare loop infiniti.


def runEnvironment(env, policy, MaxIterations, showIter=True):
    s = env.reset()
    R = 0
    for t in range(MaxIterations):
        Q_values = policy.predict(np.array([[s]]))
        a = np.argmax(Q_values[0])
        sp, r = env.step(a)
        R += r
        s = sp
        if env.StoppingCriterionSatisfied():
            break
    return R, t


def ValueIteration(env, gamma, iterations=20, convThres=1e-20):
    

    Vtable= np.zeros(env.numberOfStates())
    
    end= False # Loop Stopping criterion 
    converge= False # To know if the algorithm has converged
      
    it= 0 # Current iteration
    
    while not end: 
        auxVtable= Vtable.copy()
        
        # Update table Q(s,a)= sum_{s'} p(s'|s,a)*(r(s,a,s')+gamma*V(s))
        for s in range(env.numberOfStates()):
            Qtable= np.zeros(env.numberOfActions())
            
            for a in range(env.numberOfActions()):
                for sp in range(env.numberOfStates()):
                    
                    # Get transition probability from (s,a)->sp
                    p= env.transitionProb(s, a, sp)#* uso per calcolare la probabilità di trans 
                    if p>0.0: # Update Q-Table
                        Qtable[a]+= p*(env.rewardValue(s,a,sp) + gamma*auxVtable[sp] )#* immediata rewards           
            
            Vtable[s]= np.max(Qtable)
        
        it+= 1

        converge= np.max(np.fabs(Vtable-auxVtable)) <= convThres
        
        end= (it>=iterations or converge)
        
    return Vtable, it, converge


def ExtractPolicyFromVTable(env, Vtable, gamma):
    
    # Create policy table
    policy= np.zeros(len(Vtable), dtype=int) 
    
    
    for s in range(len(Vtable)):

        Qtable= np.zeros(env.numberOfActions())
        for a in range(env.numberOfActions()):
            for sp in range(env.numberOfStates()):
                
                # Get transition probability from (s,a)->sp
                p= env.transitionProb(s, a, sp)
                if p>0.0:
                    Qtable[a]+= p*(env.rewardValue(s,a,sp) + gamma*Vtable[sp] )
        # Update policy
        policy[s]= np.argmax(Qtable)
    return policy#politica ottimale per ogni sato ci dice quale azione intraprendere per massimizzare la  ricompensa 
#massiam futura 
#MODIFICA DELLA POLITCA GREEDY
#ORIGINALE
'''
def eGreedyPolicy(env, S, AgentPolicy, Q, epsilon):
    
    if np.random.rand() < epsilon: # Random uniform policy
        return np.random.randint(low= 0, high=env.numberOfActions()) 
    
    else:
        return AgentPolicy(env, S, Q) 
               
'''
#*MODIFICA CODICE PER PROGETTO:
    #*In questo codice epsilon_decay è un parametro che riduce epsilon ad ogni chiamata della funzione
    #*tie_break è un altro nuovo parametro che determina come gestire i casi in cui ci sono più azioni 
    #*con lo stesso valore Q massimo.
def eGreedyPolicy(env, S, AgentPolicy, Q, epsilon, epsilon_decay=0.99, tie_break="random"):
    # Decay epsilon
    epsilon *= epsilon_decay

    if np.random.rand() < epsilon: # Random policy
        if tie_break == "random":
            return np.random.randint(low=0, high=env.numberOfActions())
        elif tie_break == "first":
            # Choose the first action with the maximum Q value
            return np.argmax(Q[S])
        else:
            raise ValueError("Invalid tie_break value: {}".format(tie_break))
    else:
        return AgentPolicy(env, S, Q)
     
def AgentPolicy(env, S, Q):
    return np.argmax([Q[(S, a)] for a in range(env.numberOfActions())]) 


#^creo il modello che mi serve per la rete neurale 
def create_model(state_size, action_size):
    
    model = Sequential()
    model.add(Dense(32, input_shape=(1,)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam())
    return model


def DQLearning(env, state_size, action_size, MaxSteps, eps0, epsf, epsSteps, alpha, gamma, show=True):
    epsilon = eps0
    model = create_model(state_size, action_size)
    end = False
    it = 0
    s = env.reset()

    while not end:
        if show:
            print('Running step {}'.format(it+1))

        if np.random.rand() <= epsilon:
            a = np.random.randint(0, action_size)
        else:
            Q = model.predict(np.array([[s]]))
            a = np.argmax(Q[0])

        sp, r = env.step(a)
        r = r.squeeze()

        Q = model.predict(np.array([[s]]))
        Q_next = model.predict(np.array([[sp]]))
        Q_target = Q.copy()
        Q_target[0][a] = r + gamma * np.max(Q_next)

        model.fit(np.array([[s]]), Q_target, verbose=0)

        s = sp

        if env.StoppingCriterionSatisfied():
            s = env.reset()

        epsilon = max(epsf, eps0 + it * (epsf - eps0) / epsSteps)

        it += 1
        if it >= MaxSteps:
            end = True

    return model,  it 



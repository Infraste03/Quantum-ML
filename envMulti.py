from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, transpile
from qiskit.circuit.library import RYGate
from qiskit.circuit import Parameter
import numpy as np



# Implementation of classic MDP of the toy env in the publication
class ClassicToyEnv:
    
    # Constructor. 
    # Param MaxSteps: Maximum number of environment/agent interactions allowed for an episode
    def __init__(self, MaxSteps=np.Inf):
        
        self.MaxSteps= MaxSteps # Maximum number of steps before ending episode
        
        # ========================================
        # MDP definition
        self.nS= 4 # Number of MDP states
        self.nA= 2 # Number of MDP actions
        self.nR= 4 # Number of possible rewards

        # Transition function
        self.T= np.zeros((self.nS, self.nA, self.nS))
        self.T[0, 0, 1]= 0.7
        self.T[0, 0, 2]= 0.3
        self.T[0, 1, 1]= 1.0
        self.T[1, 1, 2]= 0.2
        self.T[1, 1, 3]= 0.8
        self.T[1, 0, 1]= 1.0
        self.T[2, 1, 0]= 0.4
        self.T[2, 1, 2]= 0.6
        self.T[2, 0, 1]= 0.2
        self.T[2, 0, 3]= 0.8
        self.T[3, 0, 2]= 1.0
        self.T[3, 1, 3]= 1.0

        # Reward function
        self.R= np.array([10, -5, 1, -10])
        
        # Current state ans step
        self.currentState= None
        self.currentStep= None
        
    
    # returns the number of states in the environment
    def numberOfStates(self):
        return self.nS
       
    # returns the number of states in the environment
    def numberOfActions(self):
        return self.nA
    
    # Returns the transition probability from state s and action a to state sp
    def transitionProb(self, s, a, sp):
        return self.T[s, a, sp]
    
    
    # Returns the reward value for the transition (s,a)->sp, i.e. r(s,a,sp)
    def rewardValue(self, s, a, sp):
        return self.R[sp]
    
        
    # Resets the environment to default state
    # Returns the current env. state
    def reset(self):
        
        # Set default env state and step number
        self.currentState= 0
        self.currentStep= 0
        return int(self.currentState)
        
    # Executes an action "action" over the environment
    # Returns the next state sp observation and reward r as  (sp, r)
    # Returns (None, None) if the Maximum number of steps criterion is True
    def step(self, action):
        
        # First: Check if the stopping criterion is True
        if self.StoppingCriterionSatisfied():
            return None, None
        
        # Calculate next state and reward
        sp= np.random.choice(range(self.nS), size=1, p=self.T[self.currentState, action].squeeze())
        r= self.R[sp]
        
        # Update step count and state
        self.currentState= sp
        self.currentStep+= 1
        
        return int(sp),r

    def StoppingCriterionSatisfied(self):
        return self.currentStep >= self.MaxSteps
    


class QuantumToyEnv(ClassicToyEnv):
    
    # Constructor. 
    # Param MaxSteps: Maximum number of environment/agent interactions allowed for an episode
    def __init__(self, MaxSteps=np.Inf):
        super().__init__(MaxSteps)
    
        # ========================================
        # Quantum MDP representation 
        self.nqS= 2 # Number of qubits for state representation= log2(nS)
        self.nqA= 1 # Number of qubits required for action representation= log2(nA)
        self.nqR= 2 # Number of qubits for reward representation= log2(nR)


        # Quantum Registers    
        self.qS= QuantumRegister(self.nqS, name='qS') # Quantum Register to store input environment states
        self.qSp= QuantumRegister(self.nqS, name='qSp') # Quantum Register to store output environment states
        self.qA= QuantumRegister(self.nqA, name='qA') # Quantum Register to store actions
        self.qR= QuantumRegister(self.nqR, name='qR') # Quantum Register to store rewards
        
        # Classical Registers
        self.cS= ClassicalRegister(self.nqS, name='cS') # Classical Register to store measured next state
        self.cR= ClassicalRegister(self.nqR, name='cR') # Classical Register to store measured reward

        # Quantum Circuit to store a cycle of an MDP
        self.qc= QuantumCircuit(self.qS, self.qA, self.qSp, self.qR, self.cS, self.cR)
        
        
        # Us implementation: Appends Us to self.qc
        self.stateParameters= [ Parameter('s'+str(i)) for i in range(self.nqS) ]
        self.__Us(self.qc, self.qS, self.stateParameters)
        

        # Ua implementation: Appends Ua to self.qc
        self.actionParameters= [ Parameter('a'+str(i)) for i in range(self.nqA) ]
        self.__Ua(self.qc, self.qA, self.actionParameters)


        # Ut implementation: Appends transition function to self.qc
        self.__Ut(self.qc, self.qS, self.qA, self.qSp)        
                

        # Ur implementation: Appends Reward function to self.qc
        self.__Ur(self.qc, self.qSp, self.qR)
        
        
        # Measurement of next state
        self.qc.measure(self.qSp, self.cS)
        
        # Measurement of reward
        self.qc.measure(self.qR, self.cR)
        
        
        # Quantum simulator
        self.sim = Aer.get_backend('qasm_simulator')
        

    def __QSampleNextState(self, probs, qc, qS, qA, qSp, control_s, control_a, control_sp= '', currentQ=0):
        
        all_controls= control_s + control_a + control_sp
        
        controlsS= []
        for i in range(len(qS)):
            controlsS.append(qS[i])
        controlsA= []
        for i in range(len(qA)):
            controlsA.append(qA[i])
        controlsSp= []
        for i in range(currentQ):
            controlsSp.append(qSp[i])
        

        if currentQ>=len(qSp):
            return
        
        else: # General case

            add_prob= np.sum(probs)
            if add_prob > 0:
                current_prob= np.sum(probs[len(probs)//2:])/add_prob
                angle= 2*np.arcsin(np.sqrt(current_prob))

                cry= RYGate(angle).control(num_ctrl_qubits=len(all_controls), 
                                             ctrl_state= all_controls)
                
                qc.append(cry, [*controlsSp, *controlsA, *controlsS, qSp[currentQ]])
        
            self.__QSampleNextState(probs[:len(probs)//2], qc, qS, qA, qSp, 
                    control_s= control_s,
                    control_a= control_a,
                    control_sp=control_sp+'0', currentQ=currentQ+1)
            self.__QSampleNextState(probs[len(probs)//2:], qc, qS, qA, qSp, 
                    control_s= control_s,
                    control_a= control_a,
                    control_sp=control_sp+'1', currentQ=currentQ+1)
    

    def __Us(self, qc, qS, params):
        for i in range(len(qS)):
            qc.rx(params[i]*np.pi, qS[i])
        qc.barrier()
        

    def __Ua(self, qc, qA, params):
        for i in range(len(qA)):
            qc.rx(params[i]*np.pi, qA[i])
        qc.barrier()
        
    
    def __Ut(self, qc, qS, qA, qSp):
        
        for s in range(self.nS): 
            for a in range(self.nA):
                
                binS= bin(s)[2:][::-1]
                while len(binS)<self.nqS:
                    binS+= '0'
                binS= binS[::-1]

                binA= bin(a)[2:][::-1]
                while len(binA)<self.nqA:
                    binA+= '0'
                binA= binA[::-1]

                probs= self.T[s, a]
                self.__QSampleNextState(probs, self.qc, self.qS, self.qA, self.qSp, 
                        control_s=binS, 
                        control_a=binA)
                self.qc.barrier()


    def __Ur(self, qc, qSp, qR):

        for i in range(len(qSp)):
            qc.cx(qSp[i], qR[i])
        self.qc.barrier()

        
    def step(self, action):

        if self.StoppingCriterionSatisfied():
            return None, None
        
        sValue= np.zeros(len(self.qS))
        counter= 0
        s= int(self.currentState)
        while s>0:
            sValue[counter]= (s & 1)
            s>>=1
            counter+= 1
        qc= self.qc.bind_parameters({self.stateParameters[i]:sValue[i] for i in range(len(self.stateParameters))})


        aValue= np.zeros(len(self.qA))
        counter= 0
        a= int(action)
        while a>0:
            aValue[counter]= (a & 1)
            a>>=1
            counter+= 1
        qc= qc.bind_parameters({self.actionParameters[i]:aValue[i] for i in range(len(self.actionParameters))})

        results = self.sim.run(transpile(qc, self.sim), shots=1).result()
        counts= results.get_counts()
        measurement= list(counts.keys())[0]
        
        binsp= measurement[:-self.nqR][::-1]
        
        sp= int(0)
        for bit in binsp:
            if bit == '1':
                sp|= 1
            sp<<= 1
        sp>>= 1 # Undo last shift
        

        binr= measurement[-self.nqR:][::-1]
    
        r= int(0)
        for bit in binr:
            if bit == '1':
                r|= 1
            r<<= 1
        r>>= 1 # Undo last shift
        r = r % len(self.R)
        

        self.currentStep+= 1
        self.currentState= sp
        
        return int(sp),r
    
#*Il codice multiagente ha diversi vantaggi rispetto a un ambiente di apprendimento automatico a singolo agente:
#*1)Simulazione di scenari più realistici: In molti scenari del mondo reale, ci sono più di un agente che interagisce 
# *con l'ambiente. Ad esempio, in un gioco multiplayer o in un sistema di traffico, ci sono molteplici agenti che
# * interagiscono tra loro. Un ambiente multiagente può simulare meglio questi scenari.
#*2)Apprendimento cooperativo e competitivo: In un ambiente multiagente, gli agenti possono apprendere non solo dalle 
# *loro interazioni con l'ambiente, ma anche dalle interazioni con altri agenti. Questo può portare a strategie di 
# *apprendimento più complesse e potenzialmente più efficaci.
#* 3) Decentralizzazione: In un ambiente multiagente, ogni agente può operare indipendentemente, 
# *il che può portare a un sistema più robusto e resiliente. Se un agente fallisce, gli altri agenti possono continuare a operare.
#*4) Parallelizzazione: In un ambiente multiagente, è possibile eseguire più azioni contemporaneamente, 
# *il che può portare a un'apprendimento più rapido.

###################################
#* ogni agente esegue un'azione in ogni passo di tempo e riceve un feedback (stato e ricompensa) indipendente. 
# *Questo permette agli agenti di apprendere in parallelo dalle loro esperienze individuali.
class MultiAgentQuantumToyEnv(QuantumToyEnv):
    def __init__(self, num_agents, MaxSteps=np.Inf):
        super().__init__(MaxSteps)
        self.num_agents = num_agents
        #num_agents= 2
        self.agent_states = [0 for _ in range(num_agents)]

    def step(self, actions):
        assert len(actions) == self.num_agents, "Number of actions must be equal to number of agents"

        states = []
        rewards = []

        for i in range(self.num_agents):
            state, reward = super().step(actions[i])
            states.append(state)
            rewards.append(reward)
            self.agent_states[i] = state

        return states, rewards
    
    

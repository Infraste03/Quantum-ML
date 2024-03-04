from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, transpile
from qiskit.circuit.library import RYGate
from qiskit.circuit import Parameter
import numpy as np
#* FILE MODIFICATO PER IL PROGETTO 

class ClassicToyEnv:

    def __init__(self, MaxSteps=np.Inf):
        
        self.MaxSteps= MaxSteps

        self.nS= 4 # Number of MDP states
        self.nA= 2 # Number of MDP actions
        self.nR= 4 # Number of possible rewards

         
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

        self.R= np.array([10, -5, 1, -10])
        
        self.currentState= None
        self.currentStep= None

    def numberOfStates(self):
        return self.nS
    
    def numberOfActions(self):
        return self.nA
    
    def transitionProb(self, s, a, sp):
        return self.T[s, a, sp]

    def rewardValue(self, s, a, sp):
        reward = self.R[sp]
        penalty_per_action = -0.01
        reward += penalty_per_action
        return reward
    
        
    def reset(self):
        

        self.currentState= 0
        self.currentStep= 0
        return int(self.currentState)
        

    def step(self, action):
        

        if self.StoppingCriterionSatisfied():
            return None, None
        
        sp= np.random.choice(range(self.nS), size=1, p=self.T[self.currentState, action].squeeze())
        r= self.R[sp]
        
        self.currentState= sp
        self.currentStep+= 1
        
        return int(sp),r

    def StoppingCriterionSatisfied(self):
        return self.currentStep >= self.MaxSteps
    
    
#************************************************************************************************
#* INIZIO MODIFICANDO SOLO L' AMBAIENTE QUANTISTICO 
#& le varie operzioni sono sono definite nei metodi __Us,__uA, __Ut e __Ur
#& posso modificare le seguenti cose :
    #& 1) Cambaire lo satate preparation operation (__Us)

class QuantumToyEnv(ClassicToyEnv):
    
    def __init__(self, MaxSteps=np.Inf):
        super().__init__(MaxSteps)

        self.nqS= 2 # Number of qubits for state representation= log2(nS)
        self.nqA= 1 # Number of qubits required for action representation= log2(nA)
        self.nqR= 2 # Number of qubits for reward representation= log2(nR)
    
        self.qS= QuantumRegister(self.nqS, name='qS') # Quantum Register to store input environment states
        self.qSp= QuantumRegister(self.nqS, name='qSp') # Quantum Register to store output environment states
        self.qA= QuantumRegister(self.nqA, name='qA') # Quantum Register to store actions
        self.qR= QuantumRegister(self.nqR, name='qR') # Quantum Register to store rewards
        

        self.cS= ClassicalRegister(self.nqS, name='cS') # Classical Register to store measured next state
        self.cR= ClassicalRegister(self.nqR, name='cR') # Classical Register to store measured reward


        self.qc= QuantumCircuit(self.qS, self.qA, self.qSp, self.qR, self.cS, self.cR)
        
        self.stateParameters= [ Parameter('s'+str(i)) for i in range(self.nqS) ]
        self.__Us(self.qc, self.qS, self.stateParameters)
        

        self.actionParameters= [ Parameter('a'+str(i)) for i in range(self.nqA) ]
        self.__Ua(self.qc, self.qA, self.actionParameters)


        self.__Ut(self.qc, self.qS, self.qA, self.qSp)        
                

        self.__Ur(self.qc, self.qSp, self.qR)
        

        self.qc.measure(self.qSp, self.cS)
        
        self.qc.measure(self.qR, self.cR)

        self.sim = Aer.get_backend('qasm_simulator')
        
    def __QSampleNextState(self, probs, qc, qS, qA, qSp, control_s, control_a, control_sp= '', currentQ=0):
        
        all_controls= control_s + control_a + control_sp
        
        # Find control qubits
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
     
     
     #& Questo metodo applica una rotazione sull asse X di ogni quibit nel registro degli stati 
     #& posso provare ad usare un quantum gate differente o una differente rotazione sull asse.
     #& per esempio provo con a ruotare sull asse y 
     #* riga orginale ==> qc.rx(params[i]*np.pi, qS[i])
     #* modifico in questa ==> qc.ry(params[i]*np.pi, qS[i])
    def __Us(self, qc, qS, params):
        for i in range(len(qS)):
            qc.ry(params[i]*np.pi, qS[i]) #^ qui ho modificato 
        qc.barrier()
        
    #& CAMBIO ANCHE ACTION PREPARATION OPERATION 
    #*del tutto simile a prima alla fine cambio anche in questo caso la rotazione 
    #* riga originale ==> qc.rx(params[i]*np.pi, qA[i])
    #* modifico in questa ==> qc.ry(params[i]*np.pi, qA[i])
    def __Ua(self, qc, qA, params):
        for i in range(len(qA)):
            qc.ry(params[i]*np.pi, qA[i]) #^ modifica qui 
        qc.barrier()
        

    #& modificare quesat è un attimo + difficile perchè alla fine dovrei andare a modificare una funnzione
    #& ricorsiva che genra un quantum circuit per il sampling del next state 
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
                
    #& posso modificare questa, 
    #& questo metodo applica un CNOT  da ogni quibit nel next state register al corrispondente qubit nel registro delle ricompense
    #* posso modificare qc.cx(qSp[i], qR[i]), applicando una differente reward operation 
    #* per esempio posso applicare la controlled Z gate 
    #* riga orginale ==> qc.cx(qSp[i], qR[i]) 
    #* ho modificato con==>  qc.cz(qSp[i], qR[i])
    def __Ur(self, qc, qSp, qR):
        
        # Hard-coded for example MDP
        for i in range(len(qSp)):
            qc.cz(qSp[i], qR[i]) #^ qui ho modificato
            
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
        
        # Post processing: Calculate the sp state as integer
        sp= int(0)
        for bit in binsp:
            if bit == '1':
                sp|= 1
            sp<<= 1
        sp>>= 1 # Undo last shift
        
        
        # Get measurement for reward
        binr= measurement[-self.nqR:][::-1]
        
        # Post processing: Calculate the actual reward value
        r= int(0)
        for bit in binr:
            if bit == '1':
                r|= 1
            r<<= 1
        r>>= 1 # Undo last shift
        r= self.R[r]
        

        self.currentStep+= 1
        self.currentState= sp
        
        return int(sp),r
    
    

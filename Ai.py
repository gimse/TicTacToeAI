import numpy as np
import mlp
import random

class Ai:
    def __init__(self,playerLetter = 'X',computerLetter = 'O'):
        try:
            file = 'q.npy'
            self.Q=np.load(file)
        except IOError:
            print('did not find file')
            self.Q=np.random.rand(np.power(3,9),9)*0.0001


        self.playerLetter=playerLetter
        self.computerLetter=computerLetter

        self.last_state=None
        self.last_action=None
        self.learningrate=0.01
        self.gamma=0.1

    def learn(self,board):
        if self.last_state!=None:
            s1=self.last_state
            a1=self.last_action
            x = self.board_to_vector(board)
            s2, x, F = find_state(x)

            qa2=np.max(self.Q[s2,:])
            self.Q[s1,a1]+=self.learningrate*(self.gamma*qa2-self.Q[s1,a1])
            #print('Q_normal',self.Q[s1,a1])
    def reward(self,reward):
        if self.last_state!=None:
            s1=self.last_state
            a1=self.last_action
            self.Q[s1,a1]+=self.learningrate*reward
            #print('Q_reward',self.Q[s1,a1])

    def move(self,board):
        x=self.board_to_vector(board)
        s,x,F = find_state(x)

        self.last_state=s

        a=self.policy(s,x)
        self.last_action=a
        if a==None:
            return 0

        v=np.zeros(len(x))
        v[a]=1
        v=F.transpose().dot(v).astype(int)
        for b in range(len(v)):
            if v[b]==1:
                return b+1

    def save(self):
        file='q.npy'
        np.save(file,self.Q)
    def policy(self,s,x):
        actions=[]
        #finding posible actions
        for i in range(len(x)):
            if x[i]==0:
                actions.append(i)
        if len(actions)==0:
            return None

        #eps=0.1
        eps = 0
        rand=random.random()
        if rand<eps:
            j=random.randint(0, len(actions)-1)
            return actions[j]
        qa=self.Q[s,actions]
        j=np.argmax(qa)
        return actions[j]

    def set_learningrate(self,learningrate):
        self.learningrate = learningrate

    def board_to_vector(self,board):
        x=np.zeros(9)
        for i in range(1,10):
            if board[i]==self.playerLetter:
                x[i-1]=1
            elif board[i]==self.computerLetter:
                x[i-1]=2
        return x
    def move_to_vector(self,move):
        t=np.zeros(9)
        t[move-1]=1
        return t


def find_state(x):
    # making rotation matrix. Rotate matrix agaist the clock
    R = np.zeros([9,9])
    R[0,2]=1
    R[1,5]=1
    R[2,8]=1
    R[3,1]=1
    R[4,4]=1
    R[5,7]=1
    R[6,0]=1
    R[7,3]=1
    R[8,6]=1
    # making swiching matric. reverse order of coloums
    S=np.zeros([9,9])
    S[0, 2] = 1
    S[1, 1] = 1
    S[2, 0] = 1
    S[3, 5] = 1
    S[4, 4] = 1
    S[5, 3] = 1
    S[6, 8] = 1
    S[7, 7] = 1
    S[8, 6] = 1
    #finding minimum s
    transforms=[np.identity(9),R,R.dot(R),R.dot(R).dot(R),S,S.dot(R),S.dot(R).dot(R),S.dot(R).dot(R).dot(R)]
    s_min=np.power(3,9) #max number
    t_min=0
    for t in range(len(transforms)):
        new_x=transforms[t].dot(x)
        s2=from_tree_numer_system(new_x)
        if s2<s_min:
            s_min=s2
            t_min=t
    s=s_min
    F=transforms[t_min].astype(int)
    x=F.dot(x).astype(int)
    return int(s),x,F


def from_tree_numer_system(x):
    s=0
    for i in range(len(x)):
        s+=x[i]*np.power(3,len(x)-1-i)
    return s

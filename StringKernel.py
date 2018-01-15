import numpy as np
from time import time

class StringKernel():
    
    def __init__(self, subseq_length, lambda_decay):
        self.n = subseq_length
        self.decay = lambda_decay
        
    def computeK(self, s, t, n):
        if min(len(s),len(t)) < n:
            return 0
        x = s[-1]
        s1 = s[:-1]
        K1sum = sum([self.computeK1(s1, t[:j], n-1) for j, char in enumerate(t) if char == x])
        return self.computeK(s1, t, n) + K1sum*self.decay**2
    
    def computeK1(self, s, t, i):
        if i == 0:
            return 1
        elif min(len(s),len(t)) < i:
            return 0
        s1 = s[:-1]
        return self.decay*self.computeK1(s1, t, i) + self.computeK2(s, t, i)
    
    def computeK2(self, s, t, i):
        if i == 0:
            return 1
        elif min(len(s),len(t)) < i:
            return 0
        idx = t.rfind(s[-1])
        if idx == len(t)-1:
            return self.decay*(self.computeK2(s, t[:-1], i) + self.decay*self.computeK1(s[:-1], t[:-1], i-1))  
        else:
            u = t[idx+1:]
            t1 = t[:idx+1]
            return self.decay**(len(u))*self.computeK2(s, t1, i)
        
    def kernel_matrix(self, docs):
        nbr_docs = len(docs)
        
        Kmatrix = np.zeros([nbr_docs, nbr_docs])
        
        for i in range(nbr_docs):
            j = i
            while(j < nbr_docs):
                Kmatrix[i,j] = self.computeK(docs[i], docs[j], self.n)
                
                Kmatrix[j,i] = Kmatrix[i,j]
                j += 1
        # Normalize Kmatrix
        diagK = np.diag(Kmatrix)
        Knorm = np.outer(diagK, diagK)
        Khatmatrix = np.true_divide(Kmatrix, np.sqrt(Knorm))
        return Khatmatrix
        

if __name__ == '__main__':
    subseq_length = 2
    decay = 0.5
    
#    docs = ['cat', 'car', 'bat', 'bar']
    docs = ['wisdom is organized life', 'science is organized knowledge', 'what are you talking about', 'who are you']
    ssk = StringKernel(subseq_length, decay)
    ts = time()
    kernel = ssk.kernel_matrix(docs)
#    Kst = ssk.computeK(docs[0], docs[1], subseq_length)
#    Kss = ssk.computeK(docs[0], docs[0], subseq_length)
#    Ktt = ssk.computeK(docs[1], docs[1], subseq_length)
    tf = time()
    print(kernel)
#    print(Kst)
#    print(Kst/np.sqrt(Kss*Ktt))
    print(tf-ts)
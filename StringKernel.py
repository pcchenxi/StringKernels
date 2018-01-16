import numpy as np
from time import time

class StringKernel():
    
    def __init__(self, substr_length, lambda_decay):
        """ String kernel constructor
        Args:
            substr_length: substring size
            lambda_decay: decay factor denoted lambda in paper
        
        """
        self.n = substr_length
        self.decay = lambda_decay
    
    def computeK(self, s, t, n):
        """ Compute K_n(s, t), according to the last line of Definition 2, described at page 424
        
        Args:
            s: first string sequence
            t: second string sequence
            n: substring length
            
        Returns:
            Recursively computed similarity measure K between s and t 
        """
        if min(len(s),len(t)) < n:
            return 0
        x = s[-1]
        s1 = s[:-1]
        K1sum = sum([self.computeK1(s1, t[:j], n-1) for j, char in enumerate(t) if char == x])
        return self.computeK(s1, t, n) + K1sum*self.decay**2
    
    def computeK1(self, s, t, i):
        """ Compute K'_i(s, t), according to the efficient computation definition, described at page 425
        
        Args:
            s: first string sequence
            t: second string sequence
            i: substring length
            
        Returns:
            Returns recursively computed auxiliary functions K' and K'' 
        """
        if i == 0:
            return 1
        elif min(len(s),len(t)) < i:
            return 0
        s1 = s[:-1]
        return self.decay*self.computeK1(s1, t, i) + self.computeK2(s, t, i)
    
    def computeK2(self, s, t, i):
        """ Compute K''_i(s, t), according to the efficient computation definition, described at page 425
        
        Args:
            s: first string sequence
            t: second string sequence
            i: substring length
            
        Returns:
            Returns recursively computed auxiliary functions K''  
        """
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
    
    def kernelMatrix(self, docs):
        """ Compute the full kernel matrix given the documents
        
        Args:
            docs: Row vector with all documents as string sequences
            
        Returns:
            Khatmatrix: Normalized matrix \hat{K}(s,t) from page 425 in paper  
        """
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
    substr_length = 2
    decay = 0.5
    show_time = True
    
    # Documents
    # docs = ['cat', 'car', 'bat', 'bar']
    docs = ['wisdom is organized life', 'science is organized knowledge', 'what are you talking about', 'who are you']
#    docs = ['wisdom is organized life', 'science is organized knowledge']

    # Computation
    ssk = StringKernel(substr_length, decay)
    start_time = time()
    kernel_matrix = ssk.kernelMatrix(docs)
    end_time = time()

    # Print results
    print 'Number of documents:', len(docs)
    print 'Documents:', docs, '\n'
    print kernel_matrix

    if (show_time):
        print '\nTime elapsed:', end_time-start_time

import numpy as np
from sklearn import svm, datasets
from sklearn import metrics
import sys
sys.path.append('../code/')
import StringKernel as ssk
from sklearn.metrics import accuracy_score
import matplotlib.pylab as plt
import re
import pickle
import time

class ExperimentRunner:
    """ Class used for conducting text document classifications
      Usage: Create instance and call run_full_test(k,lambda) or specific tests
    """
    ACQ = 0
    CORN = 1
    CRUDE = 2
    EARN = 3
    
    def show_results_table(self,precision,recall,fscore,Ktype):
        #show table with results
        
        columns = ('F1 Score', 'Precision', 'Recall')
        rows = ['Earn', 'ACQ', 'Crude', 'Corn']

        cell_text = []

        cell_text.append([fscore[ExperimentRunner.EARN],precision[ExperimentRunner.EARN],recall[ExperimentRunner.EARN]])
        cell_text.append([fscore[ExperimentRunner.ACQ],precision[ExperimentRunner.ACQ],recall[ExperimentRunner.ACQ]])
        cell_text.append([fscore[ExperimentRunner.CRUDE],precision[ExperimentRunner.CRUDE],recall[ExperimentRunner.CRUDE]])
        cell_text.append([fscore[ExperimentRunner.CORN],precision[ExperimentRunner.CORN],recall[ExperimentRunner.CORN]])

        fig=plt.figure()
        ax = plt.gca()
        the_table = ax.table(cellText=cell_text, colLabels=columns, rowLabels=rows, loc='center')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        plt.title(Ktype)

        plt.show()
        
    def prepare_data(self):
        #import data

        print "importing train + test data..."
        
        labels = {
        'acq': 0,
        'corn': 1,
        'crude': 2,
        'earn': 3
        }
        num = 1
        with open('./dataset/modapte_split_1.pkl', 'rb') as f1:
            dataset = pickle.load(f1)
        
        train_inputs = [tup[0] for tup in dataset['train'][::num]]
        train_labels = np.array([labels[tup[1]] for tup in dataset['train'][::num]])
 
        test_inputs = [tup[0] for tup in dataset['test'][::num]]
        test_labels = np.array([labels[tup[1]] for tup in dataset['test'][::num]])
        
        size = 2
        self.TrainDocVals = train_inputs[:size]
        self.TrainDocLabels = train_labels[:size]
        self.TestDocVals = test_inputs[:size]
        self.TestDocLabels = test_labels[:size]

        self.TrainDocLabels[-1] = 1

        print (len(self.TrainDocVals), len(self.TestDocVals))
        print(self.TrainDocLabels)
        print "done"
        
    def clean_doc(self, raw_doc):
        doc_text = BeautifulSoup(raw_doc, "lxml").get_text() 
    
        # Remove non-letters     
        letters_only = re.sub("[^a-zA-Z]", " ", doc_text) 
        return letters_only
    
    def __init__(self):
        self.TrainDocVals = []
        self.TrainDocLabels = []
        self.TestDocVals = []
        self.TestDocLabels = []
    
        self.prepare_data()
        
        self.WKTestGram = np.zeros((len(self.TestDocVals),len(self.TrainDocVals)))
        self.WKTrainGram = np.zeros((len(self.TrainDocVals),len(self.TrainDocVals)))

        self.NGKTestGram = np.zeros((len(self.TestDocVals),len(self.TrainDocVals)))
        self.NGKTrainGram = np.zeros((len(self.TrainDocVals),len(self.TrainDocVals)))

        self.SSKTestGram = np.zeros((len(self.TestDocVals),len(self.TrainDocVals)))
        self.SSKTrainGram = np.zeros((len(self.TrainDocVals),len(self.TrainDocVals)))
        
        self.SSKTestGramUpTo = np.zeros((15, len(self.TestDocVals),len(self.TrainDocVals)))
        self.SSKTrainGramUpTo = np.zeros((15, len(self.TrainDocVals),len(self.TrainDocVals)))

        substr_length = 2
        decay = 0.5
        self.ssk = ssk.StringKernel(substr_length, decay)

    def compute_gram_matrices(self,k=2,lamb=0.5, WK=True, NGK = True, SSK = True):
        """ Computes Kernel matrices for WK, NGK and SSK for use in tests
          Args:
            k: length (n-gram order for NGK, subsequence length for SSK)
            lamb: Decay factor for SSK
          """
        #have to reset the traingram otherwise old data makes the reflection wrong...
        self.SSKTrainGram = np.zeros((len(self.TrainDocVals),len(self.TrainDocVals)))
        
        print "computing Gram matrices"
        #compute Gram matrix for training (train,train)
        for i in xrange( 0, len(self.TrainDocVals) ):
            # if( (i+1)%10 == 0 ):
                # print "Train row %d of %d\n" % ( i+1, len(self.TrainDocVals) )       
            for j in xrange(i,len(self.TrainDocVals)):
                if(WK):
                    self.WKTrainGram[i][j] = wk.wk(self.TrainDocVals[i], self.TrainDocVals[j])
                if(NGK):
                    self.NGKTrainGram[i][j] = ngk.ngk(self.TrainDocVals[i], self.TrainDocVals[j], k)
                if(SSK):
                    time_s = time.time()
                    print (str(self.TrainDocVals[i]))
                    print (str(self.TrainDocVals[j]))
                    self.SSKTrainGram[i][j] = self.ssk.computeK(str(self.TrainDocVals[i]), str(self.TrainDocVals[j]), k)
                    time_d = time.time()
                    print ("Train row %d of %d\n" % ( i, j), time_d - time_s, self.SSKTrainGram[i][j])  
        
        #compute Gram matrix for testing (test,train). I believe this is correct due to:
        # http://stats.stackexchange.com/questions/92101/prediction-with-scikit-and-an-precomputed-kernel-svm
        for i in xrange( 0, len(self.TestDocVals) ):
            if( (i+1)%10 == 0 ):
                print "Test row %d of %d\n" % ( i+1, len(self.TestDocVals) )       
            for j in xrange(0,len(self.TrainDocVals)):
                if(WK):
                    self.WKTestGram[i][j] = wk.wk(self.TestDocVals[i], self.TrainDocVals[j])
                if(NGK):
                    self.NGKTestGram[i][j] = ngk.ngk(self.TestDocVals[i], self.TrainDocVals[j], k)
                if(SSK):
                    self.SSKTestGram[i][j] = self.ssk.computeK(str(self.TestDocVals[i]), str(self.TrainDocVals[j]), k)

        self.SSKTrainGram = self.SSKTrainGram + self.SSKTrainGram.T - np.diag(np.diag(self.SSKTrainGram))
        
        # Normalize Kmatrix
        diagK = np.diag(self.SSKTrainGram)
        Knorm = np.outer(diagK, diagK)
        self.SSKTrainGram = np.true_divide(self.SSKTrainGram, np.sqrt(Knorm))

        diagK = np.diag(self.SSKTestGram)
        Knorm = np.outer(diagK, diagK)
        self.SSKTestGram = np.true_divide(self.SSKTestGram, np.sqrt(Knorm))

        print "done"

    def do_classification(self, trainKernel, testKernel, Ktype):
        """ Runs training and prediction using given pre-computed kernels
          Args:
            trainKernel: pre-computed kernel calculated for training documents
            testKernel: pre-computed kernel calculated for test documents
          Returns:
            [precision,recall,fscore] 3 x n_categories
            precision: precision score for all categories
            recall: recall score for all categories
            fscore: F1 score for all categories
          """
        
        #TODO: run 10 times like they do in the paper?
        print "classifying for %s" %Ktype
        clf = svm.SVC(kernel='precomputed', cache_size=7000)
        
        print ("pre-train")
        clf.fit(trainKernel, self.TrainDocLabels)
        print ("pre-predic")
        label_pred = clf.predict(testKernel)
        print ("post-predict")
        precision, recall, fscore, support = metrics.precision_recall_fscore_support(self.TestDocLabels, label_pred)

        print (precision, recall, fscore, support)
        return precision, recall, fscore
        
    def run_full_test(self,k,lamb):
        """ Performs classification test with all 3 methods (WK,NGK,SSK)
          Args:
            k: length (n-gram order for NGK, subsequence length for SSK)
            lamb: Decay factor for SSK
          """
        self.compute_gram_matrices(k,lamb)
        WKres = self.do_classification(self.WKTrainGram, self.WKTestGram, 'WK')
        NGKres = self.do_classification(self.NGKTrainGram, self.NGKTestGram, 'NGK')
        SSKres = self.do_classification(self.SSKTrainGram, self.SSKTestGram, 'SSK')
        
        self.show_results_table(WKres[0],WKres[1],WKres[2],'WK')
        self.show_results_table(NGKres[0],NGKres[1],NGKres[2],'NGK')
        self.show_results_table(SSKres[0],SSKres[1],SSKres[2],'SSK')
        
    def run_WK_test(self, WKGramFileName = ""):
            """ Performs classification test with WK method
                    Args:
                       WKGramFileName: filename for pre-computed WK gram matrix (optional) 
                    Returns:
                        [precision,recall,fscore] 3 x n_categories
                        precision: precision score for all categories
                        recall: recall score for all categories
                        fscore: F1 score for all categories
              """
            if WKGramFileName != "":
                self.load_WK_Gram(WKGramFileName)
            else:
                self.compute_gram_matrices(WK=True,NGK=False, SSK=False)
                
            res = self.do_classification(self.WKTrainGram, self.WKTestGram, 'WK')
            self.show_results_table(res[0],res[1],res[2],'WK')
            
            return res
        
    def run_fast_WK_test(self):
            """ Performs classification test with WK method, using entire doc corporus for feature vectors
                    Returns:
                        [precision,recall,fscore] 3 x n_categories
                        precision: precision score for all categories
                        recall: recall score for all categories
                        fscore: F1 score for all categories
              """
            self.WKTrainGram, self.WKTestGram = wk.wkGmats(self.TrainDocVals, self.TestDocVals)
            res = self.do_classification(self.WKTrainGram, self.WKTestGram, 'WK')
            self.show_results_table(res[0],res[1],res[2],'WK')
            
            return res
        
    def run_lin_WK_test(self):
        clf = svm.SVC(kernel='linear', cache_size=7000)
        trainF, testF = wk.wkFeatVecs(self.TrainDocVals, self.TestDocVals)
        print ("pre-train")
        clf.fit(trainF, self.TrainDocLabels)
        print ("pre-predic")
        label_pred = clf.predict(testF)
        print ("post-predict")
        precision, recall, fscore, support = metrics.precision_recall_fscore_support(self.TestDocLabels, label_pred)
        self.show_results_table(precision, recall, fscore, 'WK')
        
    def run_NGK_test(self, k=2, NGKGramFileName = ""):
            """ Performs classification test with NGK method
                Args:
                    k: length (n-gram order for NGK)
                    NGKGramFileName: filename for pre-computed WK gram matrix (optional) 
                Returns:
                    [precision,recall,fscore] 3 x n_categories
                    precision: precision score for all categories
                    recall: recall score for all categories
                    fscore: F1 score for all categories    
            """
            if NGKGramFileName != "":
                self.load_NGK_Gram(NGKGramFileName)
            else:
                self.compute_gram_matrices(k=k,WK=False,NGK=True, SSK=False)
            
            res = self.do_classification(self.NGKTrainGram, self.NGKTestGram, 'NGK')
            self.show_results_table(res[0],res[1],res[2],'NGK')
            
            return res
        
    def run_fast_NGK_test(self, k=2):
            """ Performs classification test with NGK method, using entire doc corporus for feature vectors
                    Returns:
                        [precision,recall,fscore] 3 x n_categories
                        precision: precision score for all categories
                        recall: recall score for all categories
                        fscore: F1 score for all categories
              """
            self.NGKTrainGram, self.NGKTestGram = ngk.ngkGmats(self.TrainDocVals, self.TestDocVals,k)
            res = self.do_classification(self.NGKTrainGram, self.NGKTestGram, 'NGK')
            self.show_results_table(res[0],res[1],res[2],'NGK')
            
            return res
                
    def run_SSK_test(self, k=2, lamb=0.5, SSKGramFileName = ""):
            """ Performs classification test with SSK method
                Args:
                    k: length (subsequence length for SSK)
                    lamb: Decay factor for SSK
                    SSKGramFileName: filename for pre-computed WK gram matrix (optional) 
                Returns:
                    [precision,recall,fscore] 3 x n_categories
                    precision: precision score for all categories
                    recall: recall score for all categories
                    fscore: F1 score for all categories    
            """
            if SSKGramFileName != "":
                self.load_SSK_Gram(SSKGramFileName)
            else:
                self.compute_gram_matrices(k,lamb,WK=False,NGK=False, SSK=True)
            
            res = self.do_classification(self.SSKTrainGram, self.SSKTestGram, 'SSK')
            self.show_results_table(res[0],res[1],res[2],'SSK')
            
            return res
        
    def run_SSK_k_Combi_test(self, k1GramFileName, k2GramFileName):

        with open(k1GramFileName + 'Train.pickle', 'rb') as f1:
            k1TrainGram = pickle.load(f1)
        
        with open(k1GramFileName + 'Test.pickle', 'rb') as f2:
            k1TestGram = pickle.load(f2)

        with open(k2GramFileName + 'Train.pickle', 'rb') as f1:
            k2TrainGram = pickle.load(f1)
        
        with open(k2GramFileName + 'Test.pickle', 'rb') as f2:
            k2TestGram = pickle.load(f2)
            
        self.SSKTrainGram = k1TrainGram + k2TrainGram
        self.SSKTestGram = k1TestGram + k2TestGram
        
        print "done"
        res = self.do_classification(self.SSKTrainGram, self.SSKTestGram, 'SSK')
        self.show_results_table(res[0],res[1],res[2],'SSK')
        
    def run_SSK_NGK_Combi_test(self, SSKGramFileName, wSSK, wNGK):
        print "computing Gram matrices"
        #get NGK mat
        print "NGK..."
        self.NGKTrainGram, self.NGKTestGram = ngk.ngkGmats(self.TrainDocVals, self.TestDocVals,5)
        
        with open(SSKGramFileName + 'Train.pickle', 'rb') as f1:
            self.SSKTrainGram = pickle.load(f1)
        
        with open(SSKGramFileName + 'Test.pickle', 'rb') as f2:
            self.SSKTestGram = pickle.load(f2)
            
        #get SSK + NGK mat
        print "NGK + SSK..."
        
        self.NGKTestGram = self.NGKTestGram*wNGK + self.SSKTestGram*wSSK
        self.NGKTrainGram = self.NGKTrainGram*wNGK + self.SSKTrainGram*wSSK

        print "done"
        res = self.do_classification(self.NGKTrainGram, self.NGKTestGram, 'SSK + NGK')
        self.show_results_table(res[0],res[1],res[2],'SSK + NGK')
        
    def run_SSK_lam_Combi_test(self, k, lam1, lam2):
        print "computing Gram matrices"
        #compute Gram matrix for training (train,train)
        for i in xrange( 0, len(self.TrainDocVals) ):
            if( (i+1)%10 == 0 ):
                print "Train row %d of %d\n" % ( i+1, len(self.TrainDocVals) )       
            for j in xrange(0,len(self.TrainDocVals)):
                self.SSKTrainGram[i][j] = ssk.ssk(self.TrainDocVals[i], self.TrainDocVals[j], k, lam1) +                                                                 ssk.ssk(self.TrainDocVals[i], self.TrainDocVals[j], k, lam2)
        
        for i in xrange( 0, len(self.TestDocVals) ):
            if( (i+1)%10 == 0 ):
                print "Test row %d of %d\n" % ( i+1, len(self.TestDocVals) )       
            for j in xrange(0,len(self.TrainDocVals)):
                self.SSKTestGram[i][j] = ssk.ssk(self.TestDocVals[i], self.TrainDocVals[j], k, lam1) +                                                                  ssk.ssk(self.TestDocVals[i], self.TrainDocVals[j], k, lam2) 

        print "done"
        res = self.do_classification(self.SSKTrainGram, self.SSKTestGram, 'SSK')
        self.show_results_table(res[0],res[1],res[2],'SSK')
        
    def get_save_SSKMat_upto(self, k=5, lamb=0.5, SSKGramFileName = "SSKsOf"):
        """ Performs classification test with SSK method
                Args:
                    k: length (subsequence length for SSK)
                    lamb: Decay factor for SSK
                    SSKGramFileName: filename for pre-computed WK gram matrix (optional) 
                Returns:
                    [precision,recall,fscore] 3 x n_categories
                    precision: precision score for all categories
                    recall: recall score for all categories
                    fscore: F1 score for all categories    
        """

        print "computing Gram matrices"
        #compute Gram matrix for training (train,train)
        for i in xrange( 0, len(self.TrainDocVals) ):
            if( (i+1)%10 == 0 ):
                print "Train row %d of %d\n" % ( i+1, len(self.TrainDocVals) )       
            for j in xrange(i,len(self.TrainDocVals)):
                ssksUpTo = ssk.sskUpTo(str(self.TrainDocVals[i]), str(self.TrainDocVals[j]), k, lamb)
                for ki in range(0,k+1): 
                    self.SSKTrainGramUpTo[ki][i][j] = ssksUpTo[ki]
        #compute Gram matrix for testing (test,train). I believe this is correct due to:
        # http://stats.stackexchange.com/questions/92101/prediction-with-scikit-and-an-precomputed-kernel-svm
        for i in xrange( 0, len(self.TestDocVals) ):
            if( (i+1)%10 == 0 ):
                print "Test row %d of %d\n" % ( i+1, len(self.TestDocVals) )       
            for j in xrange(0,len(self.TrainDocVals)):
                ssksUpTo = ssk.sskUpTo(str(self.TestDocVals[i]), str(self.TrainDocVals[j]), k, lamb)
                for ki in range(0,k+1): 
                    self.SSKTestGramUpTo[ki][i][j] = ssksUpTo[ki]
        
        for ki in range(0,k+1): 
            refMat = self.SSKTrainGramUpTo[ki] + self.SSKTrainGramUpTo[ki].T - np.diag(np.diag(self.SSKTrainGramUpTo[ki]))
            with open(SSKGramFileName + str(ki) + 'Train.pickle', 'wb') as f1:
                pickle.dump(refMat, f1)
                                               
            with open(SSKGramFileName + str(ki) + 'Test.pickle', 'wb') as f2:
                pickle.dump(self.SSKTestGramUpTo[ki], f2)
            
        print "saved SSKs up to ", k, "in: ", SSKGramFileName
        
    #Save/load methods for use later so we don't have to recompute (might become large)
    def save_WK_Gram(self, name):
        with open(name + 'Train.pickle', 'wb') as f1:
            pickle.dump(self.WKTrainGram, f1)
        
        with open(name + 'Test.pickle', 'wb') as f2:
            pickle.dump(self.WKTestGram, f2)
        
    def load_WK_Gram(self, name):
        with open(name + 'Train.pickle', 'rb') as f1:
            self.WKTrainGram = pickle.load(f1)
        
        with open(name + 'Test.pickle', 'rb') as f2:
            self.WKTestGram = pickle.load(f2)

    def save_NGK_Gram(self, name):
        with open(name + 'Train.pickle', 'wb') as f1:
            pickle.dump(self.NGKTrainGram, f1)
        
        with open(name + 'Test.pickle', 'wb') as f2:
            pickle.dump(self.NGKTestGram, f2)
        
    def load_NGK_Gram(self, name):
        with open(name + 'Train.pickle', 'rb') as f1:
            self.NGKTrainGram = pickle.load(f1)
        
        with open(name + 'Test.pickle', 'rb') as f2:
            self.NGKTestGram = pickle.load(f2)
            
    def save_SSK_Gram(self, name):
        with open(name + 'Train.pickle', 'wb') as f1:
            pickle.dump(self.SSKTrainGram, f1)
        
        with open(name + 'Test.pickle', 'wb') as f2:
            pickle.dump(self.SSKTestGram, f2)
        
    def load_SSK_Gram(self, name):
        with open(name + 'Train.pickle', 'rb') as f1:
            self.SSKTrainGram = pickle.load(f1)
        
        with open(name + 'Test.pickle', 'rb') as f2:
            self.SSKTestGram = pickle.load(f2)


if __name__ == '__main__':
    runner = ExperimentRunner()
    runner.run_SSK_test()
    runner.save_SSK_Gram()



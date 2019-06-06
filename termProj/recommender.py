'''
@ author: 2016024793 김유진
DataScience Term Project: Predicting movie ratings
'''
import sys
import numpy as np
from src.MatrixFactorization import MatrixFactorization
from src.user import UserBasedPrediction
from src.util import *
import pickle
import gzip
import warnings
warnings.filterwarnings('ignore')
        
if __name__=="__main__":
    '''
    python recommender.py [training data] [test data]
    '''
    
    if len(sys.argv) != 3:
        print("need 2 arguments")
    else:
        base_filename = sys.argv[1]
        test_filename = sys.argv[2]
        pickle_filename = base_filename

        D = readFile(base_filename)
        A = toMat(D)
        
        pickle_filename = base_filename.split('/')[1]
        pickle_filename = 'model/' + pickle_filename.split('.')[0] + '.pkl' 
        
        try:
            with gzip.open(pickle_filename, 'rb') as f:
                factorizer = pickle.load(f)
        except:
            print('\nmodel {} not found. Start training...'.format(pickle_filename))
            factorizer = MatrixFactorization(A, k=150, learning_rate=0.01, reg_param=0.01, epochs=1000, verbose=True)
            factorizer.fit()
            with gzip.open(pickle_filename, 'wb') as f:
                pickle.dump(factorizer, f)

        factorizer.print_results()
        matrix = factorizer.get_complete_matrix()
                
        num_users,_ = A.shape
        k = int(0.3 * num_users)
        
        userBasedpredictor = UserBasedPrediction(matrix, k=k, sim='pcc')

        T = readFile(test_filename)
        count = 0
        prediction = list()
        
        for user, item, _ in T:
            i = int(user)
            j = int(item)

            p = userBasedpredictor.predict(i-1, j-1)    
            prediction.append([i, j, p])
            
            count += 1
            # if count % 1000 == 0:
            #     print("processing {}/{}".format(count, len(T)))
        
        printResult(base_filename, prediction)       

'''
TEST RESULTS
 k = 0.3 * num_users, MF epoch 800 ( u#.pkl )
+====+============+============+============+=============+
|data|     pcc    |   cosine   |  pcc w MF  | cosine w MF |
+====+============+============+============+=============+
| u1 |  0.9788111 |  0.9672778 |  0.9527246 |   0.9622327 |
| u2 |  0.9685352 |  0.9576487 |  0.9398728 |   0.9487014 |
| u3 |  0.9606577 |  0.9514656 |  0.9332089 |   0.9414392 | 
| u4 |  0.9584578 |  0.9479317 |  0.9291553 |   0.9386919 |
| u5 |  0.9588264 |  0.9475824 |  0.9341144 |   0.9413347 |

'''

        
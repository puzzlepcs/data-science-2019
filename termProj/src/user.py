'''
@ author: 2016024793 김유진
DataScience Term Project: Predicting movie ratings
- User based collaborative filtering
'''
import numpy as np
import math

class UserBasedPrediction():
    def __init__(self, matrix, k, sim):
        '''
        - data: list containing datafile from    
        '''
        self._R = matrix
        self._num_users, self._num_items = self._R.shape
        self._k = k
        self._sim = sim
        
        self._user_avg = np.array([np.mean(self._R[u][np.nonzero(self._R[u])]) for u in range(self._num_users)])
        self._user_norm = np.array([np.linalg.norm(self._R[u],ord=2) for u in range(self._num_users)])
        
        print("\ngetting {} neighbors for each user...".format(k))
        self._user_neighbors = np.array([(self.get_neighbors(u)) for u in range(self._num_users)])
        print("Done!\n")

    def pcc(self, a, i):
        '''
        Pearson correlation coefficient
        '''
        v_a = self._R[a, :] - self._user_avg[a]
        v_i = self._R[i, :] - self._user_avg[i]
        norm_a = self._user_norm[a]
        norm_i = self._user_norm[i]

        ret = np.sum((v_a) @ (v_i).T) / (norm_a * norm_i) 
        if ret < 0.:
            return 0.
        return ret

    def cosine(self, a, i):
        '''
        cosine similarity
        '''
        v_a = self._R[a, :]
        v_i = self._R[i, :]
        norm_a = self._user_norm[a]
        norm_i = self._user_norm[i]
        try:
            return np.dot(v_a,v_i) / (norm_a*norm_i)
        except:
            print("Divide by 0!")
            print("(a,i)=({},{}), norm_a: {}, norm_i: {}".format(a,i,norm_a, norm_a))
            exit() 

    def get_neighbors(self, target_user):
        '''
        get list of similarity values with other users in descending order
        using pcc similarity or cosine similarity
        - returns indices of other users and its similarity values 
          with target user.
        '''
        tmp = np.zeros(self._num_users)
        for i in range(self._num_users):
            # don't contain target_user's value
            if target_user == i:    
                continue

            # calculate similarity
            if self._sim == 'pcc':  
                sim = self.pcc(target_user, i)
            elif self._sim == 'cosine':
                sim = self.cosine(target_user, i)
            tmp[i] = sim

        # sort neighbors and similarities in descending order
        neighbors = np.flip(np.argsort(tmp))
        similarities = np.array(tmp[neighbors])
            
        return neighbors, similarities

    def predict(self, target_user, target_item):
        '''
        predict ratings of target_user and target_item
        '''
        if target_item >= self._num_items:
            return self._user_avg[target_user]
        elif target_user >= self._num_users:
            return np.mean(self._user_avg)

        neighbors, similarities = self._user_neighbors[target_user]
        
        k = list()
        p = 0
        for n, s in zip(neighbors, similarities):
            r = self._R[int(n), target_item]
            # if values of k neighbors are considered, stop 
            if len(k) == self._k:
                break
            # if rating of user n is 0, skip
            if r == 0.:     
                continue
            p += s * (r - self._user_avg[int(n)])
            k.append(s)
        p /= np.sum(k)
        p += self._user_avg[target_user]

        # if predicted rating is not properly calculated,
        # replace it with target_user's average rating value.
        if math.isnan(p) or math.isinf(p) or p < 0.:
            p = self._user_avg[target_user]      
        return p
    
    def predict_sum(self, target_user, target_item):
        neighbors, _ = self._user_neighbors[target_user]
        return  np.sum([self._R[int(n), target_item] for n in neighbors]) / self._k
    
    def predict_weighted(self, target_user, target_item):
        neighbors, similarities = self._user_neighbors[target_user]
        return np.sum([self._R[int(n), target_item] * s for n, s in zip(neighbors, similarities)]) / self._k

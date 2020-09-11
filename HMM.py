import numpy as np
import logging
logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(message)s')

class HMM(object):
    
    def __init__(self,num_latent_states,num_observation_states,**kwargs):
        self.num_latent_states = num_latent_states
        self.num_observation_states = num_observation_states
        # 初始概率分布 init_prob_dist
        if 'init_prob_dist' not in kwargs:
            self.init_prob_dist = np.random.random((self.num_latent_states,))
            self.init_prob_dist = self.init_prob_dist/np.sum(self.init_prob_dist)
        else:
            self.init_prob_dist = init_prob_dist
        # 状态转移矩阵 state_trans_matrix
        if 'state_trans_matrix' not in kwargs:
            self.state_trans_matrix = np.ones((self.num_latent_states,self.num_latent_states))
            self.state_trans_matrix = (self.state_trans_matrix.T/np.sum(self.state_trans_matrix,axis=1)).T
        else:
            self.state_trans_matrix = state_trans_matrix
        # 发射矩阵 emission_matrix
        if 'emission_matrix' not in kwargs:
            self.emission_matrix = np.random.random((self.num_latent_states,self.num_observation_states))
            self.emission_matrix = (self.emission_matrix.T/np.sum(self.emission_matrix,axis=1)).T
        else:
            self.emission_matrix = emission_matrix
        
    def forward(self,inputs):
        """
        一、目标：给定参数lambda=(init_prob_dist,state_trans_matrix,emission_matrix)，用前向算法求p(O)
        二、假设：
            1. 齐次Markov假设：p(i_t+1|o_1,...,o_t,o_t+1,i_1,...,i_t) = p(i_t+1|i_t)
            2. 观测独立性假设：p(o_t+1|o_1,...,o_t,i_1,...,i_t,i_t+1) = p(o_t+1|i_t+1)
        三、求解
        1. 标记：
            alpha_t(i) = p(o_1,...,o_t,i_t=q_i)
        2. 初始化：
            alpha_1(i) = p(o_1,i_1=q_i)
                       = p(o_1|i_1=q_i)*p(i_1=q_i)
                       = emission_matrix[q_i,o_1]*init_prob_dist[q_i]
            alpha_1 = emission_matrix[:,o_1]*init_prob_dist 
            注意：这里是position_wise乘法，不是点乘
            shape = (num_latent_state,)*(num_latent_state,) = (num_latent_state,)
        3. 迭代关系：
            alpha_t(i) = p(o_1,...,o_t,i_t=q_i)
            alpha_t+1(j) = p(o_1,...,o_t,o_t+1,i_t+1=q_j)
                         = p(o_t+1|o_1,...,o_t,i_t+1=q_j)*p(o_1,...,o_t,i_t+1=q_j)
                         = p(o_t+1|i_t+1=q_j)*sum_i(p(o_1,...,o_t,i_t=q_i,i_t+1=q_j)) #观测独立性假设
                         = p(o_t+1|i_t+1=q_j)*sum_i(p(i_t+1=q_j|o_1,...,o_t,i_t=q_i)*p(o_1,...,o_t,i_t=q_i))
                         = p(o_t+1|i_t+1=q_j)*sum_i(p(i_t+1=q_j|i_t=q_i)*p(o_1,...,o_t,i_t=q_i)) #齐次Markov假设
                         = emission_matrix[q_j,o_t+1]*sum_i(state_trans_matrix[q_i,q_j]*alpha_t(i))
            alpha_t+1 = emission_matrix[:,o_t+1]*np.dot(state_trans_matrix.T,alpha_t.T)
            注意：这里是position_wise乘法(*)，不是点乘(.*)
            shape = (num_latent_state,)*((num_latent_state,num_latent_state).*(num_latent_state,))
                  = (num_latent_state,)*(num_latent_state,)
                  = (num_latent,)
        4. 目标与alpha的关系：
            alpha_T = p(O,i_T=q_i)
            p(O) = sum_i(p(O,i_T=q_i))
                 = sum_i(alpha_T)
        """
        # 初始化
        T = len(inputs)
        alpha = []
        alpha_t = self.emission_matrix[:,inputs[0]]*self.init_prob_dist
        alpha.append(alpha_t)
        # 迭代
        for t in range(0,T-1):
            alpha_t = self.emission_matrix[:,inputs[t+1]]*np.dot(self.state_trans_matrix.T,alpha_t.T)
            alpha.append(alpha_t)
        # 返回p(O)
        p_O = np.sum(alpha_t,axis=0)
        # 返回的p_O,alpha可用于baum-welch算法
        return p_O,np.array(alpha)
        
    def backward(self,inputs):
        """
        一、目标：给定参数lambda=(init_prob_dist,state_trans_matrix,emission_matrix)，用后向算法求p(O)
        二、假设：
            1. 齐次Markov假设：p(i_t+1|o_1,...,o_t,o_t+1,i_1,...,i_t) = p(i_t+1|i_t)
            2. 观测独立性假设：p(o_t+1|o_1,...,o_t,i_1,...,i_t,i_t+1) = p(o_t+1|i_t+1)
        三、求解
        1. 标记：
            beta_t(i) = p(o_t+1,...,o_T|i_t=q_i)
        2. 初始化：
            beta_T-1(i) = p(o_T|i_T-1=q_i)
                        = sum_j(p(o_T,i_T=q_j|i_T-1=q_i))
                        = sum_j(p(o_T|i_T=q_j,i_T-1=q_i)*p(i_T=q_j|i_T-1=q_i))
                        = sum_j(p(o_T|i_T=q_j)*p(i_T=q_j|i_T-1=q_i)) #观测独立性假设
                        = sum_j(emission_matrix[q_j,o_T]*state_trans_matrix[q_i,q_j])
            beta_T-1 = np.dot(state_trans_matrix,emission_matrix[:,o_T])
            shape = (num_latent_state,num_latent_state).*(num_latent_state,)
                  = (num_latent_state,)
            注：beta_T(i) = [1,...,1]   shape=(num_latent_state,)
        3. 迭代关系：
            beta_t+1(j) = p(o_t+2,...,o_T|i_t+1=q_j)
            beta_t(i) = p(o_t+1,...,o_T|i_t=q_i)
                      = sum_j(p(o_t+1,...,o_T,i_t+1=q_j|i_t=q_i))
                      = sum_j(p(o_t+1|o_t+2,...,o_T,i_t+1=q_j,i_t=q_i)*p(o_t+2,...,o_T,i_t+1=q_j|i_t=q_i))
                      = sum_j(p(o_t+1|i_t+1=q_j)*p(o_t+2,...,o_T|i_t=q_i,i_t+1=q_j)*p(i_t+1=q_j|i_t=q_i)) #观测独立性假设
                      = sum_j(p(o_t+1|i_t+1=q_j)*p(o_t+2,...,o_T|i_t+1=q_j)*p(i_t+1=q_j|i_t=q_i)) #概率图阻断原理：p(o_t+2,...,o_T|i_t=q_i,i_t+1=q_j) = p(o_t+2,...,o_T|i_t+1=q_j)
                      = sum_j(emission_matrix[q_j,o_t+1]*beta_t+1(j)*state_trans_matrix[q_i,q_j])
            beta_t = np.dot(beta_t+1,(emission_matrix[:,o_t+1]*state_trans_matrix).T)
            注意：这里是position_wise乘法(*)，不是点乘(.*)
            shape = np.sum((num_latent_states,)*(num_latent_states,)*(num_latent_states,num_latent_states),axis=0)
                  = np.sum((num_latent_states,num_latent_states),axis=0)
                  = (num_latent_states,)
        4. 目标与beta的关系：
            beta_1 = p(o_2,...,o_T|i_1=q_i)
            p(O) = sum_i(p(O,i_1=q_i))
                 = sum_i(p(O|i_1=q_i)*p(i_1=q_i))
                 = sum_i(p(o_1,o_2,...,o_T|i_1=q_i)*p(i_1=q_i))
                 = sum_i(p(o_1|o_2,...,o_T,i_1=q_i)*p(o_2,...,o_T|i_1=q_i)*p(i_1=q_i))
                 = sum_i(p(o_1|i_1=q_i))*p(o_2,...,o_T|i_1=q_i)*p(i_1=q_i)) #观测独立性假设
                 = sum_i(emission_matrix[q_i,o_1]*beta_1*init_prob_dist[q_i])
                 = np.dot(beta_1,emission_matrix[:,o_1]*init_prob_dist[q_i])
        """
        # 初始化
        T = len(inputs)
        beta = []
        beta_T = np.array([1.]*self.num_latent_states)
        beta.append(beta_T)
        beta_t = np.dot(self.state_trans_matrix,self.emission_matrix[:,inputs[-1]])
        beta.append(beta_t)
        # 迭代
        for t in range(0,T-2)[::-1]:
            beta_t = np.sum(self.state_trans_matrix*beta_t*self.emission_matrix[:,inputs[t+1]],axis=1)
            beta.append(beta_t)
        p_O = sum(beta_t*self.init_prob_dist*self.emission_matrix[:,inputs[0]])
        # 返回的p_O,beta可用于baum-welch算法
        return p_O,np.array(beta[::-1])
    
    def train(self,inputs,conv_loss=1e-8):
        """
        一、目标：利用Baum-Welch算法无监督训练HMM
        二、迭代公式（EM算法）：
            lambda_t+1 = argmax_lambda sum_I[log(p(O,I|lambda))*p(I|O,lambda_t)]
                       = argmax_lambda sum_I[log(p(I,O|lambda))*p(I,O|lambda_t)/p(O|lambda_t)]
                       = argmax_lambda sum_I[log(p(I,O|lambda))*p(I,O|lambda_t)]
            lambda_t+1 = (init_prob_dist_t+1,state_trans_matrix_t+1,emission_matrix_t+1)
            设Q函数为：
                Q(lambda,lambda_t) = sum_I[log(p(I,O|lambda))*p(I,O|lambda_t)]
            由于：
                #假设序列长为T
                p(O|lambda) = sum_I[p(O,I|lambda)]
                            = sum_i_1[sum_i_2[...[sum_i_T(init_prob_dist[i_1]*prod_i(state_trans_matrix[i_t-1,i_t])*prod_i(emission_matrix[i_t,o_t]))]]]
            所以Q函数为：
                Q(lambda,lambda_t) = sum_I[(log(init_prob_dist[i_1])+sum_i(log(state_trans_matrix[i_t-1,i_t]))+sum_i(log(emission_matrix[i_t,o_t])))*p(O,I|lambda_t)]
            分别拆分为：
                init_prob_dist_t+1 = argmax_init_prob_dist sum_I[log(init_prob_dist[i_1])*p(O,I|lambda_t)]
                state_trans_matrix_t+1 = argmax_state_trans_matrix sum_I[sum_i(log(state_trans_matrix[i_t-1,i_t]))*p(O,I|lambda_t)]
                emission_matrix_t+1 = argmax_emission_matrix sum_I[sum_i(log(emission_matrix[i_t,o_t]))*p(O,I|lambda_t)]
                这里涉及到约束优化问题：
                    np.sum(init_prob_dist_i) = 1
                    np.sum(state_trans_matrix,axis=0) = [1,...,1]     shape=(state_trans_matrix.shape[0],)
                    np.sum(emission_matrix,axis=0) = [1,...,1]     shape=(emission_matrix.shape[0],)
                需要用到拉格朗日法求解，这里不做过多展开，具体请参照李航《统计学习方法》的章节《隐马尔可夫模型》：Baum-Welch算法
            求得各迭代公式可以用李航《统计学习方法》中给出的公式：
                gamma_t(i) = p(i_t=q_i|O)
                           = p(i_t=q_i,O)/p(O)
                           = alpha_t(i)*beta_t(i)/sum_j(alpha_t(j)*beta_t(j))
                gamma = alpha*beta/np.sum(alpha*beta,axis=1)
                      = alpha*beta/p(O) # np.sum(alpha*beta,axis=1)这个值实际上就是p(O)值
                xi_t(i,j) = alpha_t(i)*state_trans_matrix[q_i,q_j]*emission_matrix[q_j,o_t+1]*beta_t+1(j)/sum_i[sum_j(alpha_t(i)*state_trans_matrix[q_i,q_j]*emission_matrix[q_j,o_t+1]*beta_t+1(j))]
                xi_t = (alpha_t*state_trans_matrix.T).T*emission_matrix[:,o_t+1]*beta_t+1/np.sum((alpha_t*state_trans_matrix.T).T*emission_matrix[:,o_t+1]*beta_t+1)
            1. init_prob_dist：
                init_prob_dist[q_i] = gamma_1(i)
                init_prob_dist = gamma_1
            2. state_trans_matrix：
                state_trans_matrix[q_i,q_j] = sum_(t=1~T-1)[xi_t(i,j)]/sum_(t=1~T-1)[gamma_t(i)]
                                            = np.sum(xi(i,j),axis=0) / np.sum(gamma[:-1](i),axis=0)
                state_trans_matrix = np.sum(xi,axis=0)/np.sum(gamma[:-1],axis=0)
            3. emission_matrix:
                emission_matrix[q_j,o_t=v_k] = sum_(t=1~T,o_t=v_k)[gamma_t(j)]/sum_(t=1~T)[gamma_t(j)]
                # 这里o_t=v_k暂时还没想好怎么直接用numpy处理，暂时只能用循环处理
                
                #emission_from_gamma.shape = (num_latent_states,num_observation_states)
                emission_from_gamma = np.zeros((num_latent_states,num_observation_states))
                for gamma_i,o_i in zip(gamma,o):
                    #gamma_i.shape = (1,num_latent_states)
                    emission_from_gamma[:,o_i] += gamma_i 
                emission_matrix = (emission_from_gamma.T/np.sum([gamma_1,gamma_2,...,gamma_T],axis=0)).T
        """
        logger = logging.getLogger('Baum-Welch')
        epochs = 1
        l2_loss = np.Inf
        while True:
            
            init_prob_dist = np.zeros((self.num_latent_states,))
            state_trans_matrix = np.zeros((self.num_latent_states,self.num_latent_states))
            emission_matrix = np.zeros((self.num_latent_states,self.num_observation_states))
            
            for input_item in inputs:
                p_O,alpha = self.forward(input_item)
                p_O,beta = self.backward(input_item)
                gamma = alpha*beta/p_O
                xi = []
                for t in range(len(input_item)-1):
                    xi_t = (alpha[t]*self.state_trans_matrix.T).T*self.emission_matrix[:,input_item[t+1]]*beta[t+1]/np.sum((alpha[t]*self.state_trans_matrix.T).T*self.emission_matrix[:,input_item[t+1]]*beta[t+1])
                    xi.append(xi_t)
                xi = np.array(xi)
                #计算init_prob_dist
                init_prob_dist += gamma[0]
                #计算state_trans_matrix
                state_trans_matrix += (np.sum(xi,axis=0).T/np.sum(gamma[:-1],axis=0)).T
                #计算emission_matrix
                emission_from_gamma = np.zeros((self.num_latent_states,self.num_observation_states))
                for gamma_i,o_i in zip(gamma,input_item):
                    emission_from_gamma[:,o_i] += gamma_i
                emission_matrix += (emission_from_gamma.T/np.sum(gamma,axis=0)).T
            init_prob_dist /= len(inputs)
            state_trans_matrix /= len(inputs)
            emission_matrix /= len(inputs)
            
            l2_loss_old = l2_loss
            l2_loss = np.sum(np.power(init_prob_dist-self.init_prob_dist,2))+np.sum(np.power(state_trans_matrix-self.state_trans_matrix,2))+np.sum(np.power(emission_matrix-self.emission_matrix,2))
            if l2_loss_old-l2_loss <= 0 or l2_loss <= conv_loss:
                logger.info('training finished!')
                break
            logger.info('epochs:{}\tloss:{}'.format(epochs,l2_loss))
            self.init_prob_dist = init_prob_dist
            self.state_trans_matrix = state_trans_matrix
            self.emission_matrix = emission_matrix
            epochs += 1
        return
    
    def decode(self,inputs):
        """
        一、目标：利用Viterbi算法解码
        二、Viterbi算法：
            1. 初始化：
                delta_1(i) = init_prob_dist[q_i]*emission_matrix[q_i,o_1]  --> delta_1 = init_prob_dist*emission_matrix[:,o_1]
                psi_1(i) = 0
            2. 迭代公式：
                对于t=2,3,...,T
                    delta_t(i) = max_(1<=j<=num_latent_states)[[delta_t-1(j)*state_trans_matrix[q_j,q_i]]*emission_matrix[q_i,o_t]]
                    psi_t(i) = argmax_(1<=j<=num_latent_states)[delta_t-1(j)*state_trans_matrix[q_j,q_i]]  # 这个式子可以带上最后一项emission_matrix[q_i,o_t]]，argmax和最后一项没关系
            3. 终止：
                P* = max_(1<=i<=num_latent_states)[delta_T(i)]
                i*_T = argmax_(1<=i<=num_latent_states)[delta_T(i)]
            4. 最优路径回溯
                i*_t = psi_t+1(i*_t+1)
        """
        logger = logging.getLogger('Viterbi')
        logger.info('start decoding...')
        delta,psi,route = [],[],[]
        # 初始化
        delta_1 = self.init_prob_dist*self.emission_matrix[:,inputs[0]]
        psi_1 = np.zeros((self.num_latent_states,))
        delta.append(delta_1)
        psi.append(psi_1)
        # 迭代
        for t in range(1,len(inputs)):
            iter_func = (delta[-1] * self.state_trans_matrix.T).T*self.emission_matrix[:,inputs[t]]
            delta_t = np.max(iter_func,axis=0)
            psi_t = np.argmax(iter_func,axis=0)
            delta.append(delta_t)
            psi.append(psi_t)
        # 最优路径回溯
        route_T = np.argmax(delta[-1])
        route.append(route_T)
        for t in range(len(inputs)-1)[::-1]:
            route_t = psi[t+1][route[-1]]
            route.append(route_t)
        route = route[::-1]
        logger.info('decoding finished!')
        return route

if __name__=="__main__":
    '''
    # 初始化状态概率向量
    init_prob_dist = np.array((0.2,0.4,0.4))
    # 状态转移矩阵
    state_trans_matrix = np.array([[0.5,0.2,0.3],[0.3,0.5,0.2],[0.2,0.3,0.5]])
    # 发射矩阵
    emission_matrix = np.array([[0.5,0.5],[0.4,0.6],[0.7,0.3]])

    # 初始化隐马尔可夫参数
    hmm = HMM(num_latent_states=3,num_observation_states=2,init_prob_dist=init_prob_dist,state_trans_matrix=state_trans_matrix,emission_matrix=emission_matrix)
    # 输入状态序列
    inputs = [0,1,0]

    # 前向算法
    p_O,alpha = hmm.forward(inputs)
    print('p(O):',p_O)
    print('alpha:\n',alpha)

    # 后向算法
    p_O,beta = hmm.backward(inputs)
    print('p(O):',p_O)
    print('beta:\n',beta)
    '''
    '''
    # 随机初始化学习
    hmm = HMM(num_latent_states=3,num_observation_states=2)

    # 给定训练序列
    hmm.train([[0,1,0],[1,1,0,1,0,0,0,1]])

    # 输出三参数
    print('init_prob_dist:\n',hmm.init_prob_dist)
    print('state_trans_matrix:\n',hmm.state_trans_matrix)
    print('emission_matrix:\n',hmm.emission_matrix)
    '''
    # 初始化状态概率向量
    init_prob_dist = np.array((0.2,0.4,0.4))
    # 状态转移矩阵
    state_trans_matrix = np.array([[0.5,0.2,0.3],[0.3,0.5,0.2],[0.2,0.3,0.5]])
    # 发射矩阵
    emission_matrix = np.array([[0.5,0.5],[0.4,0.6],[0.7,0.3]])

    # 初始化隐马尔可夫参数
    hmm = HMM(num_latent_states=3,num_observation_states=2,init_prob_dist=init_prob_dist,state_trans_matrix=state_trans_matrix,emission_matrix=emission_matrix)

    # 给定训练序列
    hmm.train([[0,1,0],[1,1,0,1,0,0,0,1]])

    # 输出三参数
    print('init_prob_dist:\n',hmm.init_prob_dist)
    print('state_trans_matrix:\n',hmm.state_trans_matrix)
    print('emission_matrix:\n',hmm.emission_matrix)
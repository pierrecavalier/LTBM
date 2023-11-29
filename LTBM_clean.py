import numpy as np
from scipy.special import digamma, gammaln

import time






class LTBM:
    def __init__(self, A, W, dictionnary, alpha, Q, L, K, tol=1e-3, YX=None, seed=2023) :
        """
        @param A: matrice d'incidence
        @param W: dict ; clé: (i, j), ou A[i, j] != 0 ; valeur: liste de liste de mots (! encodés en id 0, 1,... !)
        @param dictionnary: list ; liste de tous les mots
        @param alpha: vecteur des paramètres de concentration de Dirichlet de thehta (fixé dans le modele)
        
        @param Q: nombre de cluster ligne
        @param L: nombre de cluster colonne
        @param K: nombre de sujets

        """
        np.random.seed(seed)

        self.tol = tol

        self.A = A
        self.T = [(i, j) for i, j in zip(*np.where(A != 0))]  # non-zero entries
        self.W = W

        self.V = len(dictionnary)

        self.Q = Q
        self.L = L
        self.K = K

        self.alpha = alpha
        self.gln_alpha = gammaln(self.alpha)
        self.gln_alpha_sum = gammaln(np.sum(self.alpha))

        if not len(alpha) == self.K : 
            raise ValueError("alpha should be a vector of size K")

        if YX is None :
            self._init_XY()
        else :
            self.Y, self.X = YX
        self._init_params()

        self.lower_bound = -np.inf

    def fit(self, max_iter=20, max_iterVEM=100, verbose=1):

        greedysucces = True
        for iter in range(max_iter):
            lb0 = self.lower_bound
            time0 = time.time()

            # Variational EM
            self.VEM(max_iterVEM, verbose=verbose>1)
            timeVEM = time.time() - time0
            lbVEM = self.lower_bound
            deltaVEM = self.lower_bound - lb0

            if not greedysucces and deltaVEM < self.tol:
                if verbose:
                    print('\nIteration {}: {:.3f} (deltaVEM: {:.4f})'.format(iter + 1, self.lower_bound, deltaVEM))
                    print('VEM: {:.2f}s\n'.format(timeVEM))
                break


            # Greedy search
            self.greedy_search_XY()
            timeGreedy = time.time() - timeVEM - time0
            deltaGreedy = self.lower_bound - lbVEM
            
            if deltaGreedy > self.tol : greedysucces = True
            else : greedysucces = False
        
            if verbose>0:
                print('\nIteration {}: {:.3f} (deltaVEM: {:.4f}, deltaGreedy: {:.4f})'.format(iter + 1, self.lower_bound, deltaVEM, deltaGreedy))
                print('VEM: {:.2f}s, Greedy: {:.2f}s\n'.format(timeVEM, timeGreedy))
            if not greedysucces and deltaVEM < self.tol:
                break

    def A_clust(self):
        Y_clust = np.argmax(self.Y, axis=1)
        X_clust = np.argmax(self.X, axis=1)
        A_clust = np.zeros(self.A.shape)

        for i, j in self.T:
            A_clust[i, j] = Y_clust[i] + X_clust[j] * self.Q + 1

        return A_clust


    def _init_XY(self):
        self.Y = np.zeros((self.A.shape[0], self.Q))
        for i in range(self.A.shape[0]):
            self.Y[i, np.random.randint(0, self.Q)] = 1

        self.X = np.zeros((self.A.shape[1], self.L))
        for j in range(self.A.shape[1]):
            self.X[j, np.random.randint(0, self.L)] = 1



    def _init_params(self) : 

        self._update_rho_delta()
        self._update_pi()
        self.beta = np.random.dirichlet(np.ones(self.V), self.K) 

        ### variational parameters

        # parametrise la distribution q(theta) (Dirichlet distrib)
        self.gamma = np.ones((self.Q, self.L, self.K))
        self.dg_gamma = digamma(self.gamma) - digamma(np.sum(self.gamma, axis=2))[:, :, np.newaxis]

        # parametrise la distribution q(Z) (multinomial distrib)
        self.phi = [[None for _ in range(self.A.shape[1])] for _ in range(self.A.shape[0])] 

        for i, j in self.T:
            l = []
            for doc in self.W[i][j]:
                l.append(np.ones((len(doc), self.K)) / self.K)

            self.phi[i][j] = l


    def VEM(self, max_iter=100, verbose=True):

        for i in range(max_iter):
            old_lb = self.lower_bound

            # E-step
            self._update_phi()
            self._update_gamma()


            # M-step
            self._update_rho_delta
            self._update_pi()
            self._update_beta()

            # Compute lower bound
            self.compute_lower_bound()

            delta = self.lower_bound - old_lb
            if verbose:
                print('Iteration {}: {:.3f} (delta: {:.4f})'.format(i + 1, self.lower_bound, delta))
            if delta < self.tol:
                break



    def _update_rho_delta(self):
        self.rho = np.sum(self.Y, axis=0) / self.A.shape[0]
        self.delta = np.sum(self.X, axis=0) / self.A.shape[1]

    def _update_pi(self):
        self.pi = np.zeros((self.Q, self.L))
        for q, l in np.ndindex(self.Q, self.L):
            in_clus_ql = self.Y[:, np.newaxis, q] @ self.X[np.newaxis, :, l]
            self.pi[q, l] = np.sum(in_clus_ql * self.A) / np.sum(in_clus_ql)

    def _update_beta(self):
        self.beta = np.zeros((self.K, self.V))
        for i, j in self.T:
            for doc, phi in zip(self.W[i][j], self.phi[i][j]):
                for v, p in zip(doc, phi):
                    for k in range(self.K):
                        self.beta[k, v] += p[k]

        if not np.all(self.beta>0):
            print('Warning: beta has non positive values')
            self.beta += 1e-10

        self.beta = self.beta / np.sum(self.beta, axis=1)[:, np.newaxis]
            
        self.ln_beta = np.log(self.beta)


        # print('beta', self.beta.shape, np.sum(self.beta, axis=1))


    def _update_phi(self):
        for i, j in self.T:
            q = np.where(self.Y[i, :] == 1)[0][0]
            l = np.where(self.X[j, :] == 1)[0][0]

            for k in range(self.K):
                exp_comp = np.exp(self.dg_gamma[q, l, k])

                for doc, phi in zip(self.W[i][j], self.phi[i][j]):
                    for v, p in zip(doc, phi):
                        p[k] = self.beta[k, v] * exp_comp

            # Normalize over topics
            for phi in self.phi[i][j] :
                if not np.all(phi>0):
                    print('Warning: phi has non positive values')
                    phi += [p + 1e-10 for p in phi]
                
                for idx, p in enumerate(phi):
                    phi[idx] = p / np.sum(p)


    def _update_gamma(self):

        self.gamma = np.ones((self.Q, self.L, self.K)) * self.alpha
        for i, j in self.T:
            q = np.where(self.Y[i, :] == 1)[0][0]
            l = np.where(self.X[j, :] == 1)[0][0]

            for k in range(self.K):
                self.gamma[q, l, k] += sum(self.phi[i][j][d][n][k] for d in range(len(self.W[i][j])) for n in range(len(self.W[i][j][d])))

        self.dg_gamma = digamma(self.gamma) - digamma(np.sum(self.gamma, axis=2))[:, :, np.newaxis]
        self.gln_gamma = gammaln(self.gamma)
        self.gln_gamma_sum = gammaln(np.sum(self.gamma, axis=2))


    def compute_lower_bound(self):
        """
        Compute the variational lower bound
        """
        lb = 0

        #### L(q(.) | A, Y, X, beta) (refer to Appendix C) ###
        for i, j in self.T:
            q = np.where(self.Y[i, :] == 1)[0][0]
            l = np.where(self.X[j, :] == 1)[0][0]

            for doc, phi in zip(self.W[i][j], self.phi[i][j]):
                for v, p in zip(doc, phi):
                    for k in range(self.K):

                        # 1-st term
                        lb += p[k] * self.ln_beta[k, v]

                        # 2-nd term
                        lb += p[k] * self.dg_gamma[q, l, k]

                        # 4-rd term
                        lb -= p[k] * np.log(p[k])
        
        for q, l in np.ndindex(self.Q, self.L):
            # 3-rd term
            lb += self.gln_alpha_sum - np.sum(self.gln_alpha) + np.sum( (self.alpha-1) * self.dg_gamma[q, l, :])

            # 5-th term
            lb -= self.gln_gamma_sum[q, l] - np.sum(self.gln_gamma[q, l, :]) + np.sum((self.gamma[q, l, :]-1) * self.dg_gamma[q, l, :])

        #### p(A, Y, X | pi, rho, delta) (eq 15) ###
        for i, j in self.T:
            q = np.where(self.Y[i, :] == 1)[0][0]
            l = np.where(self.X[j, :] == 1)[0][0]

            lb += np.log(self.pi[q, l] * self.rho[q] * self.delta[l])

        # print('lb', lb)
        
        self.lower_bound = lb


    def greedy_search_XY(self) :
        qs_num = np.sum(self.Y, axis=0)
        for i in range(self.A.shape[0]) :            
            q = np.argmax(self.Y[i, :])

            if qs_num[q] > 1 :
                current_lb = self.lower_bound
                best_lb = current_lb
                q_final = q

                self.Y[i, q] = 0
                
                for q_try in range(self.Q) :
                    if q_try == q : continue

                    self.Y[i, q_try] = 1
                    self.compute_lower_bound()

                    if self.lower_bound > best_lb :
                        q_final = q_try
                        best_lb = self.lower_bound

                    self.Y[i, q_try] = 0

                self.Y[i, q_final] = 1
                self.lower_bound = best_lb

                qs_num[q] -= 1
                qs_num[q_final] += 1
        
        ls_num = np.sum(self.X, axis=0)
        for j in range(self.A.shape[1]) :            
            l = np.argmax(self.X[j, :])

            if ls_num[l] > 1 :
                current_lb = self.lower_bound
                best_lb = current_lb
                l_final = l

                self.X[j, l] = 0
                
                for l_try in range(self.L) :
                    if l_try == l : continue

                    self.X[j, l_try] = 1
                    self.compute_lower_bound()

                    if self.lower_bound > best_lb :
                        l_final = l_try
                        best_lb = self.lower_bound

                    self.X[j, l_try] = 0

                self.X[j, l_final] = 1
                self.lower_bound = best_lb

                ls_num[l] -= 1
                ls_num[l_final] += 1





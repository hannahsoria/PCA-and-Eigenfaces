'''pca_cov.py
Performs principal component analysis using the covariance matrix approach
YOUR NAME HERE
CS 251/2 Data Analysis Visualization
Spring 2023
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans as km

from PIL import Image
from numpy import asarray


class PCA_COV:
    '''
    Perform and store principal component analysis results
    '''

    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: pandas DataFrame. shape=(num_samps, num_vars)
            Contains all the data samples and variables in a dataset. Should be set as an instance variable.
        '''
        self.data = data

        # vars: Python list. len(vars) = num_selected_vars
        #   String variable names selected from the DataFrame to run PCA on.
        #   num_selected_vars <= num_vars
        self.vars = None

        # A: ndarray. shape=(num_samps, num_selected_vars)
        #   Matrix of data selected for PCA
        self.A = None

        # normalized: boolean.
        #   Whether data matrix (A) is normalized by self.pca
        self.normalized = None

        # A_proj: ndarray. shape=(num_samps, num_pcs_to_keep)
        #   Matrix of PCA projected data
        self.A_proj = None

        # e_vals: ndarray. shape=(num_pcs,)
        #   Full set of eigenvalues (ordered large-to-small)
        self.e_vals = None
        # e_vecs: ndarray. shape=(num_selected_vars, num_pcs)
        #   Full set of eigenvectors, corresponding to eigenvalues ordered large-to-small
        self.e_vecs = None

        # prop_var: Python list. len(prop_var) = num_pcs
        #   Proportion variance accounted for by the PCs (ordered large-to-small)
        self.prop_var = None

        # cum_var: Python list. len(cum_var) = num_pcs
        #   Cumulative proportion variance accounted for by the PCs (ordered large-to-small)
        self.cum_var = None

        # normalization variables
        self.mins = None
        self.maxs = None
        self.ranges = None
        self.means = None

    def get_prop_var(self):
        '''(No changes should be needed)'''
        return self.prop_var

    def get_cum_var(self):
        '''(No changes should be needed)'''
        return self.cum_var

    def get_eigenvalues(self):
        '''(No changes should be needed)'''
        return self.e_vals

    def get_eigenvectors(self):
        '''(No changes should be needed)'''
        return self.e_vecs

    def covariance_matrix(self, data):
        '''Computes the covariance matrix of `data`

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_vars)
            `data` is NOT centered coming in, you should do that here.

        Returns:
        -----------
        ndarray. shape=(num_vars, num_vars)
            The covariance matrix of centered `data`

        NOTE: You should do this wihout any loops
        NOTE: np.cov is off-limits here — compute it from "scratch"!
        '''
        center = lambda x: x - x.mean() # center the data
        data = center(data) # center
        N = data.shape[0] # number of rows
        data -= data.mean(axis=0) # get the mean of all of the columns
        total = N - 1 # one less than num of rows
        cov_mat = np.dot(data.T, data.conj()) / total #complex conjugate of a complex number is the number with an equal real part and an imaginary part equal in magnitude but opposite in sign
        return(cov_mat)



    def compute_prop_var(self, e_vals):
        '''Computes the proportion variance accounted for by the principal components (PCs).

        Parameters:
        -----------
        e_vals: ndarray. shape=(num_pcs,)

        Returns:
        -----------
        Python list. len = num_pcs
            Proportion variance accounted for by the PCs
        '''
        total_e = np.sum(e_vals) # total of all eigen values
        pcs = [] # create new list
        for i in range(len(e_vals)): # for every eigen value
            pc = e_vals[i] / total_e # divide value by the total
            pcs.append(pc) # add it to the list
        return pcs


    def compute_cum_var(self, prop_var):
        '''Computes the cumulative variance accounted for by the principal components (PCs).

        Parameters:
        -----------
        prop_var: Python list. len(prop_var) = num_pcs
            Proportion variance accounted for by the PCs, ordered largest-to-smallest
            [Output of self.compute_prop_var()]

        Returns:
        -----------
        Python list. len = num_pcs
            Cumulative variance accounted for by the PCs
        '''
        accum_var = [] # create new list
        var = 0 # variable set to zero to be accumulated
        for i in range(len(prop_var)): # for all of the proportional variables
            var = var + prop_var[i] # add each prop variable to the accumulated variable
            accum_var.append(var) # append to the list each loop
        return accum_var

    def pca(self, vars, normalize=False):
        '''Performs PCA on the data variables `vars`

        Parameters:
        -----------
        vars: Python list of strings. len(vars) = num_selected_vars
            1+ variable names selected to perform PCA on.
            Variable names must match those used in the `self.data` DataFrame.
        normalize: boolean.
            If True, normalize each data variable so that the values range from 0 to 1.

        NOTE: Leverage other methods in this class as much as possible to do computations.

        TODO:
        - Select the relevant data (corresponding to `vars`) from the data pandas DataFrame
        then convert to numpy ndarray for forthcoming calculations.
        - If `normalize` is True, normalize the selected data so that each variable (column)
        ranges from 0 to 1 (i.e. normalize based on the dynamic range of each variable).
            - Before normalizing, create instance variables containing information that would be
            needed to "undo" or reverse the normalization on the selected data.
        - Make sure to compute everything needed to set all instance variables defined in constructor,
        except for self.A_proj (this will happen later).
        '''
        #set variables
        self.vars = vars
        self.A = self.data[self.vars]
        self.A = self.A.to_numpy()

        # save normalization data
        self.mins = np.min(self.A, axis = 0)
        self.maxs = np.max(self.A, axis = 0)
        self.means = np.mean(self.A, axis = 0)
        self.ranges = self.maxs - self.mins

        if normalize == True:
            self.A = (self.A - self.mins) / self.ranges #normalization computation

        #set variables
        self.e_vals, self.e_vecs = np.linalg.eig(self.covariance_matrix(self.A))
        self.e_vals = np.real(self.e_vals)
        self.e_vecs = np.real(self.e_vecs)
        self.prop_var = self.compute_prop_var(self.e_vals)
        self.cum_var = self.compute_cum_var(self.prop_var)

    def elbow_plot(self, num_pcs_to_keep=None):
        '''Plots a curve of the cumulative variance accounted for by the top `num_pcs_to_keep` PCs.
        x axis corresponds to top PCs included (large-to-small order)
        y axis corresponds to proportion variance accounted for

        Parameters:
        -----------
        num_pcs_to_keep: int. Show the variance accounted for by this many top PCs.
            If num_pcs_to_keep is None, show variance accounted for by ALL the PCs (the default).

        NOTE: Make plot markers at each point. Enlarge them so that they look obvious.
        NOTE: Reminder to create useful x and y axis labels.
        NOTE: Don't write plt.show() in this method
        '''
        pv = self.prop_var # list of cummulative variances
        if num_pcs_to_keep == None:
            k = self.e_vals.size # all of the cummulative variances
        else:
            k = num_pcs_to_keep # just the PC specified
            
        plt.figure(figsize=(16,8)) # figure
        plt.plot(range(k),self.cum_var[:k], marker=".", markersize=20) # x = top PCs included y = proportion variance accounted for
        plt.xlabel('PCs')
        plt.ylabel('Proportion Variance')
        plt.title('The Elbow Method showing Cumulative Variance')



    def pca_project(self, pcs_to_keep):
        '''Project the data onto `pcs_to_keep` PCs (not necessarily contiguous)

        Parameters:
        -----------
        pcs_to_keep: Python list of ints. len(pcs_to_keep) = num_pcs_to_keep
            Project the data onto these PCs.
            NOTE: This LIST contains indices of PCs to project the data onto, they are NOT necessarily
            contiguous.
            Example 1: [0, 2] would mean project on the 1st and 3rd largest PCs.
            Example 2: [0, 1] would mean project on the two largest PCs.

        Returns
        -----------
        pca_proj: ndarray. shape=(num_samps, num_pcs_to_keep).
            e.g. if pcs_to_keep = [0, 1],
            then pca_proj[:, 0] are x values, pca_proj[:, 1] are y values.

        NOTE: This method should set the variable `self.A_proj`
        '''
        Ac = self.A - self.A.mean(axis=0) # subtract the means
        pca_proj = Ac @ self.e_vecs[:,pcs_to_keep] # multiply the subtracted meas and the selected eigen vectors
        self.A_proj = pca_proj

        return pca_proj

    def pca_then_project_back(self, top_k):
        '''Project the data into PCA space (on `top_k` PCs) then project it back to the data space

        Parameters:
        -----------
        top_k: int. Project the data onto this many top PCs.

        Returns:
        -----------
        ndarray. shape=(num_samps, num_selected_vars)

        TODO:
        - Project the data on the `top_k` PCs (assume PCA has already been performed).
        - Project this PCA-transformed data back to the original data space
        - If you normalized, remember to rescale the data projected back to the original data space.
        '''
        k_val = [i for i in range(top_k)] # make a list 
        pca_back = self.pca_project(k_val) # project
        pca_back = pca_back @ self.e_vecs[:,:top_k].T + self.means # multiply the matrix by all vectors transposed and add the means
        if self.normalized == True: 
            pca_back = (pca_back * self.ranges) + self.mins # denormalize
        
        return pca_back
    

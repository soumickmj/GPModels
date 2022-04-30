
import numpy as np
import os


"""
This is the main module for:
1-testing if there is any healthy brain images
2-save the results of our code


"""

"""
__author__ = "Hadya Yassin"
Created on Thu March 20 15:00:00 2021

"""

class Helper(object):


    def healthy_brains(self, mt):


        """# Check the whole dataset for Healthy brain images:
        ***Crucial in 2D models***
        """

        #healthy brain counter initialization
        j = 0


        # Check the whole dataset b4 splitting
        for m in mt:

            #m is a np array of size (512,512) with values (0-255) in list mt

            #For 2D models classification, healthy brains (empty masks) should be checked
            #And the labels should be tweeked accordingly if healthy label = healthy, if not label = label
            if np.amax(m) == 0:  #=no tumor=healthy brain
                #count how many healthy ones
                j = j + 1

        print("there are {} empty masks (healthy brains) out of 3064 dataset total samples".format(j))


        # incase of the presence of healthy brains, set num_classes = 4
        if j >= 1:
            num_class = 4

        # if not then set it to 3 labels
        else:
            num_class = 3

        return j, num_class


   

    def Save_Results(self, type, value1, value2, main_path, m_name, orient, batch, aug, epochs, i, b):

        """# Function to save the results values in text file
        """
    
        print("Saving Results value in a .md file")

        filename = (main_path + '/Results/{}_{}_{}b_{}a_{}e.md'.format(m_name[1], orient, batch, aug, epochs))
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        if type == "Train":
            with open(filename, "a") as output:
                output.write("\n" + "Epoch is {}, ".format(i+1) + "total loops {}".format(b+1) + "batch size {}".format(batch) + "\n")
                output.write("{}_acc = ".format(type) + str(value1) + "\n")
                output.write("last_{}_loss = ".format(type) + str(value2))

        elif type == "CR & JI": 
            with open(filename, "a") as output:
                output.write("\n" + "Classification report is : " + "\n" + str(value1) + "\n")
                output.write("Jaccard index = " + str(value2))

        else: 
            with open(filename, "a") as output:
                output.write("\n" + "{}_acc = ".format(type) + str(value1) + "\n")
                output.write("last_{}_loss = ".format(type) + str(value2))




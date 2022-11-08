
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

        print(
            f"there are {j} empty masks (healthy brains) out of 3064 dataset total samples"
        )



        # incase of the presence of healthy brains, set num_classes = 4
        num_class = 4 if j >= 1 else 3
        return j, num_class


   

    def Save_Results(self, type, value1, value2, main_path, m_name, orient, batch, aug, epochs, i, b):

        """# Function to save the results values in text file
        """

        print("Saving Results value in a .md file")

        filename = f'{main_path}/Results/{m_name[1]}_{orient}_{batch}b_{aug}a_{epochs}e.md'

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        if type == "Train":
            with open(filename, "a") as output:
                output.write(
                    "\n"
                    + f"Epoch is {i + 1}, "
                    + f"total loops {b + 1}"
                    + f"batch size {batch}"
                    + "\n"
                )

                output.write(f"{type}_acc = {str(value1)}" + "\n")
                output.write(f"last_{type}_loss = {str(value2)}")

        elif type == "CR & JI": 
            with open(filename, "a") as output:
                output.write("\n" + "Classification report is : " + "\n" + str(value1) + "\n")
                output.write(f"Jaccard index = {str(value2)}")

        else: 
            with open(filename, "a") as output:
                output.write("\n" + f"{type}_acc = " + str(value1) + "\n")
                output.write(f"last_{type}_loss = {str(value2)}")




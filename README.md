# ERA_HW6_CNN
CNN 
     # MNIST Assignment

     This repository contains the implementation of a neural network model for the MNIST dataset.

     ## Model Details

     - **Total Parameter Count:** 19,818
     - **Batch Normalization:** Yes
     - **DropOut:** Yes
     - **Fully Connected Layer or GAP:** Yes

     ## Tests

     - **Total Parameter Count Test:** Pass (less than 20,000)
     - **Test Accuracy:** Pass (greater than 99.4%)


------------CALCULATIONS COPIED FROM EXCEL-----------------------------------
Layer	Nin	Rin	Jin	S	p	Nout	Rout	Jout	Input Channel	Output channel	K	Parameters
Conv1	28	1	1	1	1	28	3	1	1	64	3	576									
Conv2	28	3	1	1	1	28	5	1	64	16	3	9216
Max pool	28	5	1	2	1	14	6	2	16	16	2	8
Conv3	14	6	2	1	1	14	10	2	16	16	3	2304
Conv4	14	10	2	1	1	14	14	2	16	16	3	2304
Max pool	14	14	2	2	1	7	15	4	16	16	2	8
Conv 5	7	15	4	1	1	7	23	4	16	32	3	4608												
FC									32	10		320
												
										Total Parameter Count		19344

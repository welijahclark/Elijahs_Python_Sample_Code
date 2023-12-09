#Raw Assignment 4 Code
#1.e version 1
x = "Flying"
y = " Circus"
print("Monty Python's", x+y)

#1.e version 2
x = "Flying"
y = "Circus"
print("Monty Python's", x+y)

#1.f
#1.e version 1

#1.g
G_Answer = (11 % 3)**4
print(G_Answer)

#2.a
two_a = 11.0
isinstance(two_a, float)
two_a_two = int(11.0)
type(two_a_two)

#2.b
((12-3)/3) == 2
((12-3)/3) != 2

#2.c
A = 3.0
B = 5
C = 15.5
D = "Stats"

print(A)
print(B)
print(A+B)
print(type(C))
print(type(D))

#2.d
a = "STATISTICS"
b = a.lower()
print(a)
print(b)

#2.e
k = "statistic"
h = list(k)
h.append("s")
print(k)
print(h)
print(k.count("s"))
print(h.count("s"))

#2.f
L = [1, 2, 3, 5, 8, 13, 21, 34]
Slice_1 = L[2:5]
print(Slice_1)
Slice_2 = L[1:6:2]
print(Slice_2)
Slice_3 = L[6:2:-1]
print(Slice_3)

#2.g
tuple_time = (2, 3, 5, 7, 11, 13, 17, 19)
print ("The length of this tuple is", len(tuple_time))
print ("The maximum value in this tuple is", max(tuple_time))
print ("The sum of all items in this tuple is", sum(tuple_time))

#2.h
wec_dict = { "Game" : ['Candyland', 'Scrabble'],
             "Time" : [25, 135],
             "Difficulty" : ['easy', 'moderate']
           }
print(wec_dict)
wec_dict["Year"] = [1948, 1931]


#Note: This is just a line for graphical leading. I would like to optimize the typographical layout of the results here.
print("       ")

print(wec_dict)

#Note: This is just a line for graphical leading. I would like to optimize the typographical layout of the results here.
print("       ")

wec_dict.get("Game")


#Importing numpy library in advance here
import numpy as np

#3.a
array_1 = [5,6,3,2,4,7,8,9]
array_2 = [7,1,7,8,6,5,8,11]
array_3 = [5,6,-3,4,4,9,1,0]

array_a = np.array([array_1,array_2,array_3])
print(array_a)

#Note: This is just a line for graphical leading. I would like to optimize the typographical layout of the results here.
print("       ")

array_a2 = np.reshape(a=array_a, newshape = (4,6))
print(array_a2)


#3.b
array_b1 = np.arange(-3, 4, 1)
print(array_b1)

#Note: This is just a line for graphical leading. I would like to optimize the typographical layout of the results here.
print("       ")

array_b2 = np.random.rand(1,7)
print(array_b2)

#Note: This is just a line for graphical leading. I would like to optimize the typographical layout of the results here.
print("       ")

array_b3 = array_b - array_b2
print(array_b3)

array_b1 = np.arange(-3, 10, 1)
print(array_b)


#3.c
array_c = np.random.rand(9,4)
print(array_c)

#Note: This is just a line for graphical leading. I would like to optimize the print results here.
print("       ")

array_c2 = array_c[3:8, 2:]
print(array_c2)

#3.d
array_x = np.array([12, 24, 32, 19])
array_mu = np.array([20, 20, 20, 20])
array_sigma = np.array([5, 5, 5, 5])

array_top = np.array(array_x - array_mu)
array_zed = np.array(array_top/array_sigma)
print(array_zed)

#3.e
arr1 = np.random.normal(loc=40, scale = 10, size = 50)
arr2 = np.random.normal(loc=25, scale = 10, size = 50)
print("The mean of array one is", np.mean(arr1))
print("The standard deviation of array two is", np.std(arr1))

#Note: This is just a line for graphical leading. I would like to optimize the typographical layout of the results here.
print("       ")

print("The mean of array two is", np.mean(arr2))
print("The standard deviation of array two is", np.std(arr2))

#Note: This is just a line for graphical leading. I would like to optimize the typographical layout of the results here.
print("       ")

array_e = arr1 + arr2
print(array_e)

#Note: This is just a line for graphical leading. I would like to optimize the typographical layout of the results here.
print("       ")

print("The mean of array e is", np.mean(array_e))
print("The standard deviation of array e is", np.std(array_e))


#importing Seaborn here
import seaborn as sns

#3.f
arr1 = np.random.normal(loc=40, scale = 10, size = 50)
arr2 = np.random.normal(loc=25, scale = 10, size = 50)
array_e = arr1 + arr2

sns.histplot(array_e, kde = True)